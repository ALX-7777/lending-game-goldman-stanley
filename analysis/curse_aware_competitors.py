"""Curse-aware competitors: instead of rejecting or netting the risky band, they lend to it at
model rate + a LARGE margin M (10 to 40 points) to cover the winner's curse.
(1) Is that profitable for them?  (2) What is our best response: keep the net, or lend in the band at
model + margin too (with or without the net as a cap)?
Run from the project folder after run_round1.py:  python analysis/curse_aware_competitors.py"""
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
tau_arr = np.array([chosen["tau"][s] for s in seg])
m_arr = np.array([chosen["margins"][s] for s in seg])


def curse_aware(p, M, switch=None, cap=None):
    """Below the switch (our thresholds by default): break-even + our segment margins.
    Above: break-even + M, capped at 100%; optionally no offer above PD `cap`."""
    sw = tau_arr if switch is None else np.full(n, switch)
    r = np.where(p <= sw, p / (1 - p) + m_arr, np.minimum(p / (1 - p) + M, 1.0))
    if cap is not None:
        r = np.where(p > cap, np.inf, r)
    return r


def ours(band_rule):
    """Our rates: chosen rule below the threshold, `band_rule` (array) inside the band."""
    return np.where(p2 > tau_arr, band_rule, p2 / (1 - p2) + m_arr)


be2 = p2 / (1 - p2)
frac = np.clip((p2 - tau_arr) / np.maximum(0.25 - tau_arr, 1e-9), 0, 1)
our_variants = {"net 0.828 (chosen)": ours(np.full(n, NET)),
                "ramp 0.70 -> 0.828": ours(0.70 + (NET - 0.70) * frac)}
for M in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    our_variants[f"band: model + {M:.2f}"] = ours(np.minimum(be2 + M, 1.0))
for M in [0.10, 0.20, 0.30]:
    our_variants[f"band: min(model + {M:.2f}, 0.828)"] = ours(np.minimum(be2 + M, NET))

# ---------------------------------------------------------------- competitor cases
base = ARCHETYPES["base"]
rb1, rb3 = price(p1, seg=seg, **base), price(p3, seg=seg, **base)
cases = {"base / base": (rb1, rb3)}
for M in [0.10, 0.15, 0.20, 0.30, 0.40]:
    cases[f"both curse-aware, model + {M:.2f} in band"] = (curse_aware(p1, M), curse_aware(p3, M))
for M in [0.10, 0.20, 0.30]:
    cases[f"both curse-aware above 25%, model + {M:.2f}"] = (curse_aware(p1, M, switch=0.25), curse_aware(p3, M, switch=0.25))
cases["curse-aware +0.20 / base"] = (curse_aware(p1, 0.20), rb3)
cases["curse-aware +0.20 / clone 0.898"] = (curse_aware(p1, 0.20), clone_rates(p3, seg, chosen, 0.898))
ev = FastEvaluator(cases, n, seeds, pF)

print("=== 1. Our expected profit (M EUR) against curse-aware competitors, by our treatment of the band ===")
T = pd.DataFrame({name: {k: np.mean(v) / 1e6 for k, v in ev.profits(r2).items()} for name, r2 in our_variants.items()}).T
print(T.round(2).to_string())
print("\nbest response per case:\n" + pd.DataFrame({"best": T.idxmax(), "profit": T.max().round(2),
                                                    "net 0.828": T.loc["net 0.828 (chosen)"].round(2)}).to_string())

# ---------------------------------------------------------------- 2. do curse-aware competitors make money in the band?
print("\n=== 2. Competitors' realized profit (M EUR, total and inside THEIR band) when both are curse-aware and we keep the net ===")
r2 = our_variants["net 0.828 (chosen)"]
rows = []
for M in [0.10, 0.15, 0.20, 0.30, 0.40]:
    r1, r3 = cases[f"both curse-aware, model + {M:.2f} in band"]
    acc = []
    for seed in seeds:
        res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
        won1 = (choice == 0) & (p1 > tau_arr)
        acc.append({"lender 1 total": res[0]["profit_realized"] / 1e6, "lender 1 band loans": won1.sum(),
                    "lender 1 band default": y[won1].mean(), "lender 1 band rate": r1[won1].mean(),
                    "lender 1 band profit": np.where(y[won1] == 1, -LOAN, LOAN * r1[won1]).sum() / 1e6,
                    "our total": res[US]["profit_realized"] / 1e6, "our share": res[US]["share"]})
    rows.append(pd.DataFrame(acc).mean().rename(f"model + {M:.2f}"))
print(pd.DataFrame(rows).round(3).to_string())

# ---------------------------------------------------------------- 3. and if we answer with model + margin too: everyone's profit
print("\n=== 3. Both competitors curse-aware at model + 0.20; our profit and theirs for each of our responses ===")
r1, r3 = cases["both curse-aware, model + 0.20 in band"]
rows = []
for name in ["net 0.828 (chosen)", "band: model + 0.10", "band: model + 0.15", "band: model + 0.20", "band: model + 0.30",
             "band: min(model + 0.20, 0.828)"]:
    r2 = our_variants[name]
    acc = []
    for seed in seeds:
        res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
        won = (choice == US) & (p2 > tau_arr)
        acc.append({"our total": res[US]["profit_realized"] / 1e6, "our band loans": won.sum(),
                    "our band default": y[won].mean() if won.any() else np.nan,
                    "our band profit": np.where(y[won] == 1, -LOAN, LOAN * r2[won]).sum() / 1e6,
                    "lender 1 total": res[0]["profit_realized"] / 1e6, "lender 3 total": res[2]["profit_realized"] / 1e6})
    rows.append(pd.DataFrame(acc).mean().rename(name))
print(pd.DataFrame(rows).round(3).to_string())
