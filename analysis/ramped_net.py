"""Should the safety net be flat (0.828 for everyone above the threshold) or should it increase with the
predicted PD, i.e. 'model prediction + an increasing margin' inside the risky band, e.g. 78% for the least
risky of them and 82.8% for the riskiest?  A lower rate only wins a customer when a competitor's offer sits
between the two rates, so we measure, world by world, how many band customers each variant actually wins
and what they cost.
Run from the project folder after run_round1.py:  python analysis/ramped_net.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lending_game.competitors import ARCHETYPES, clone_rates, competitor_rates, price  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import LOAN, US, FastEvaluator, draw_types, simulate  # noqa: E402
from lending_game.strategy import Rule  # noqa: E402

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
rule = Rule(p2, seg, seg)
tau_arr = np.array([chosen["tau"][s] for s in seg])
in_band = p2 > tau_arr
r_normal = rule.rates({**chosen, "tail_rate": None})  # inf inside the band


def ramped(L, top=NET, p_top=0.25):
    """Net rising linearly from L at the threshold to `top` at PD = p_top (flat above); flat if L == top."""
    frac = np.clip((p2 - tau_arr) / np.maximum(p_top - tau_arr, 1e-9), 0, 1)
    net = L + (top - L) * frac
    return np.where(in_band, net, r_normal)


variants = {"flat 0.828 (chosen)": ramped(NET), "flat 0.78": ramped(0.78, 0.78), "flat 0.70": ramped(0.70, 0.70),
            "ramp 0.78 -> 0.828": ramped(0.78), "ramp 0.70 -> 0.828": ramped(0.70),
            "ramp 0.60 -> 0.828": ramped(0.60), "ramp 0.50 -> 0.828": ramped(0.50), "ramp 0.40 -> 0.828": ramped(0.40)}

# ---------------------------------------------------------------- competitor cases (same families as clone_scenarios.py)
base, norej = ARCHETYPES["base"], ARCHETYPES["no_reject"]
rb1, rb3 = price(p1, seg=seg, **base), price(p3, seg=seg, **base)
cases = {"nobody nets: base/base": (rb1, rb3),
         "nobody nets: no_reject/no_reject": (price(p1, seg=seg, **norej), price(p3, seg=seg, **norej))}
for X in [0.99, 0.95, 0.90, 0.85]:
    cases[f"base(25%) both net {X:.2f}"] = (price(p1, seg=seg, net=X, **base), price(p3, seg=seg, net=X, **base))
for X in [0.989, 0.95, 0.92, 0.90, 0.898, 0.89, 0.87, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    cases[f"both clones net {X:.3f}"] = (clone_rates(p1, seg, chosen, X), clone_rates(p3, seg, chosen, X))
for X in [0.95, 0.898, 0.85, 0.75]:
    cases[f"clone {X:.3f} / base"] = (clone_rates(p1, seg, chosen, X), rb3)
for X in [0.95, 0.898, 0.85]:
    cases[f"moderate clones (tau x2) net {X:.3f}"] = (clone_rates(p1, seg, chosen, X, 2.0), clone_rates(p3, seg, chosen, X, 2.0))
families = {"nobody nets": [c for c in cases if c.startswith("nobody")],
            "base nets (above 25%)": [c for c in cases if c.startswith("base(")],
            "moderate clones": [c for c in cases if c.startswith("moderate")],
            "one clone / base": [c for c in cases if c.startswith("clone")],
            "both clones": [c for c in cases if c.startswith("both clones")]}
priors = {"skeptical": [0.70, 0.15, 0.05, 0.07, 0.03], "moderate": [0.50, 0.20, 0.10, 0.12, 0.08],
          "paranoid": [0.30, 0.20, 0.15, 0.15, 0.20]}
ev = FastEvaluator(cases, n, seeds, pF)

print("=== 1. Expected profit (M EUR): mean by family of competitor cases, and prior-weighted ===")
rows = []
for name, r2 in variants.items():
    res = pd.Series({k: np.mean(v) / 1e6 for k, v in ev.profits(r2).items()})
    row = {fam: res[cols].mean() for fam, cols in families.items()}
    for pname, probs in priors.items():
        row[f"prior: {pname}"] = sum(pr * res[cols].mean() for pr, cols in zip(probs, families.values()))
    rows.append(pd.Series(row, name=name))
print(pd.DataFrame(rows).round(2).to_string())

# ---------------------------------------------------------------- customers won in the band, world by world
print("\n=== 2. Customers won INSIDE the risky band (above our threshold), realized default and profit, by world ===")
worlds = {"nobody nets (base/base)": cases["nobody nets: base/base"],
          "nobody rejects (no_reject/no_reject)": cases["nobody nets: no_reject/no_reject"],
          "both clones net 0.85": cases["both clones net 0.850"],
          "both clones net 0.80": cases["both clones net 0.800"]}
for wname, (r1, r3) in worlds.items():
    rows = []
    for vname in ["flat 0.828 (chosen)", "flat 0.78", "ramp 0.78 -> 0.828", "ramp 0.60 -> 0.828", "ramp 0.40 -> 0.828"]:
        r2 = variants[vname]
        acc = []
        for seed in seeds:
            res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
            won = (choice == US) & in_band
            acc.append({"band loans won": won.sum(), "mean rate": r2[won].mean() if won.any() else np.nan,
                        "realized default": y[won].mean() if won.any() else np.nan,
                        "band profit (M)": np.where(y[won] == 1, -LOAN, LOAN * r2[won]).sum() / 1e6})
        rows.append(pd.DataFrame(acc).mean().rename(vname))
    print(f"\n--- {wname} ---  (band = {in_band.sum()} applicants)")
    print(pd.DataFrame(rows).round(3).to_string())
