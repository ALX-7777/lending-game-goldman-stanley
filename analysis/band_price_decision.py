"""The price of the risky band is a bet between worlds: cheap (model + 3%) loses everywhere, medium
(model + 10-20 points) wins against curse-aware margin-adders, high (net 0.828) wins against rejecters
and netters.  Weigh the candidates over all competitor families with explicit priors.
Run from the project folder after run_round1.py:  python analysis/band_price_decision.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lending_game.competitors import ARCHETYPES, clone_rates, competitor_rates, price  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import FastEvaluator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "output" / "round1"

past = pd.read_csv(HERE / "PastLoans.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
chosen = strategy["rule"]
NET = chosen["tail_rate"]
seeds = [0, 1, 2]
tau_arr = np.array([chosen["tau"][s] for s in seg])
m_arr = np.array([chosen["margins"][s] for s in seg])
be2 = p2 / (1 - p2)


def curse_aware(p, M, switch=None):
    sw = tau_arr if switch is None else np.full(n, switch)
    return np.where(p <= sw, p / (1 - p) + m_arr, np.minimum(p / (1 - p) + M, 1.0))


def ours(band_rule):
    return np.where(p2 > tau_arr, band_rule, be2 + m_arr)


frac = np.clip((p2 - tau_arr) / np.maximum(0.25 - tau_arr, 1e-9), 0, 1)
variants = {"net 0.828 (chosen)": ours(np.full(n, NET)),
            "ramp 0.70 -> 0.828": ours(0.70 + (NET - 0.70) * frac),
            "band: model + 0.10": ours(np.minimum(be2 + 0.10, 1.0)),
            "band: model + 0.15": ours(np.minimum(be2 + 0.15, 1.0)),
            "band: model + 0.20": ours(np.minimum(be2 + 0.20, 1.0)),
            "hybrid: model + 0.15 if PD <= 0.18, net above": ours(np.where(p2 <= 0.18, be2 + 0.15, NET)),
            "hybrid: model + 0.15 if PD <= 0.15, net above": ours(np.where(p2 <= 0.15, be2 + 0.15, NET))}

# ---------------------------------------------------------------- competitor cases by family
base, norej = ARCHETYPES["base"], ARCHETYPES["no_reject"]
rb1, rb3 = price(p1, seg=seg, **base), price(p3, seg=seg, **base)
orig = competitor_rates(p1, p3, seg)
cases, fam = {}, {}


def add(family, name, pair):
    cases[name] = pair
    fam.setdefault(family, []).append(name)


for k in ["base / base", "no_reject / no_reject", "aggressive / aggressive", "slide / slide", "loose / tight"]:
    add("lend normally (various margins)", k, orig[k])
add("reject the band", "cautious / cautious", orig["cautious / cautious"])
add("reject the band", "cautious / slide", orig["cautious / slide"])
for X in [0.99, 0.95, 0.90]:
    add("net above 25%", f"base both net {X:.2f}", (price(p1, seg=seg, net=X, **base), price(p3, seg=seg, net=X, **base)))
add("net above 25%", "smart / smart", orig["smart / smart"])
for X in [0.989, 0.95, 0.898, 0.85, 0.80, 0.75]:
    add("clones (net the band)", f"both clones net {X:.3f}", (clone_rates(p1, seg, chosen, X), clone_rates(p3, seg, chosen, X)))
for X in [0.95, 0.898, 0.85]:
    add("one clone / base", f"clone {X:.3f} / base", (clone_rates(p1, seg, chosen, X), rb3))
for M in [0.10, 0.15, 0.20, 0.30, 0.40]:
    add("curse-aware: band at model + M", f"both curse-aware +{M:.2f}", (curse_aware(p1, M), curse_aware(p3, M)))
for M in [0.10, 0.20, 0.30]:
    add("curse-aware above 25% only", f"both curse-aware(25%) +{M:.2f}", (curse_aware(p1, M, 0.25), curse_aware(p3, M, 0.25)))
add("mixed", "curse-aware +0.20 / base", (curse_aware(p1, 0.20), rb3))
add("mixed", "curse-aware +0.20 / clone 0.898", (curse_aware(p1, 0.20), clone_rates(p3, seg, chosen, 0.898)))
add("mixed", "curse-aware +0.20 / cautious", (curse_aware(p1, 0.20), price(p3, seg=seg, **ARCHETYPES["cautious"])))

ev = FastEvaluator(cases, n, seeds, pF)
R = pd.DataFrame({v: {k: np.mean(x) / 1e6 for k, x in ev.profits(r2).items()} for v, r2 in variants.items()})

print("=== 1. Mean expected profit (M EUR) by family of competitor behaviour ===")
F = pd.DataFrame({f: R.loc[cols].mean() for f, cols in fam.items()}).T
print(F.round(2).to_string())

print("\n=== 2. Prior-weighted (M EUR). Family probabilities: ===")
priors = {  # lend normally, reject band, net above 25%, clones, one clone, curse-aware band, curse-aware 25%, mixed
    "skeptical (clones and big margins both rare)": [0.55, 0.10, 0.10, 0.03, 0.05, 0.08, 0.05, 0.04],
    "moderate":                                     [0.40, 0.10, 0.10, 0.07, 0.08, 0.12, 0.07, 0.06],
    "your view (rational margin-adders likely)":    [0.30, 0.08, 0.08, 0.05, 0.06, 0.25, 0.10, 0.08],
    "net-heavy (clones likely)":                    [0.30, 0.08, 0.10, 0.20, 0.10, 0.10, 0.06, 0.06],
}
for name, w in priors.items():
    print(f"  {name}: " + ", ".join(f"{f}={p:.0%}" for f, p in zip(fam, w)))
D = pd.DataFrame({name: sum(p * F.loc[f] for f, p in zip(fam, w)) for name, w in priors.items()})
print()
print(D.round(2).to_string())
print("\nbest variant per prior:\n" + D.idxmax().to_string())
