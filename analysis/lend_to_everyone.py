"""What happens if we trust the model and lend to everyone at break-even + margin, with no threshold
and no safety net -- including the very risky applicants?
Run from the project folder after run_round1.py:  python analysis/lend_to_everyone.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lending_game.competitors import clone_rates, competitor_rates, price  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import LOAN, US, draw_types, simulate  # noqa: E402
from lending_game.strategy import Rule  # noqa: E402
from run_round1 import evaluate_strategy  # noqa: E402

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
rule = Rule(p2, seg, seg)
seeds = [0, 1, 2]

comp = competitor_rates(p1, p3, seg)
comp["both clones net 0.898"] = (clone_rates(p1, seg, chosen, 0.898), clone_rates(p3, seg, chosen, 0.898))
comp["both clones net 0.70"] = (clone_rates(p1, seg, chosen, 0.70), clone_rates(p3, seg, chosen, 0.70))


def everyone(margin_by_seg, extra_risky=0.0):
    """Lend to everyone: break-even + segment margin (+ extra margin for PD > 20%), never net, never refuse."""
    prm = copy.deepcopy(chosen)
    prm["tau"] = {s: 1.0 for s in prm["tau"]}
    prm["margins"] = margin_by_seg
    r = rule.rates(prm)
    return np.minimum(r + extra_risky * (p2 > 0.20), 1.0)


strategies = {
    "A. everyone at break-even + 3%": price(p2, margin=0.03),
    "B. everyone, our segment margins (3-4%)": everyone(chosen["margins"]),
    "C. everyone, segment margins, +5% extra if PD > 20%": everyone(chosen["margins"], 0.05),
    "D. everyone, segment margins, +15% extra if PD > 20%": everyone(chosen["margins"], 0.15),
    "E. CHOSEN rule (thresholds + net 0.828)": rule.rates(chosen),
}

print("=== Realized profit (M EUR) by scenario; mean over 3 type draws ===")
tabs = {}
for name, r2 in strategies.items():
    tabs[name] = evaluate_strategy(r2, comp, p2, pF, y, seeds)
P = pd.DataFrame({name: t["profit_real_M"] for name, t in tabs.items()})
print(P.round(2).to_string())
print("\n=== Summary ===")
S = pd.DataFrame({"mean (12 scenarios)": P.iloc[:12].mean(), "worst (12 scenarios)": P.iloc[:12].min(),
                  "vs clones 0.898": P.loc["both clones net 0.898"], "vs clones 0.70": P.loc["both clones net 0.70"],
                  "mean share": pd.Series({k: t["share"].iloc[:12].mean() for k, t in tabs.items()}),
                  "realized default on won (base)": pd.Series({k: t.loc["base / base", "realized_default"] for k, t in tabs.items()}),
                  "predicted (base)": pd.Series({k: t.loc["base / base", "our_p2_on_won"] for k, t in tabs.items()})})
print(S.round(3).to_string())

# ---------------------------------------------------------------- where does 'lend to everyone' lose money?
print("\n=== Strategy B (everyone, segment margins) in the base scenario, by band of OUR predicted PD ===")
r2 = strategies["B. everyone, our segment margins (3-4%)"]
r1, r3 = comp["base / base"]
acc = []
for seed in seeds:
    res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
    won = choice == US
    df = pd.DataFrame({"band": pd.cut(p2, [0, .05, .10, .15, .20, .25, .30, .40, 1.0]), "won": won, "p2": p2, "y": y,
                       "rate": r2, "profit": np.where(won, np.where(y == 1, -LOAN, LOAN * r2), 0.0)})
    g = df.groupby("band", observed=True)
    w = df[won].groupby("band", observed=True)
    acc.append(pd.DataFrame({"applicants": g.size(), "share won": g["won"].mean(), "our rate": w["rate"].mean(),
                             "predicted PD on won": w["p2"].mean(), "realized default on won": w["y"].mean(),
                             "break-even rate for realized": w["y"].mean() / (1 - w["y"].mean()),
                             "profit (M)": g["profit"].sum() / 1e6}))
band = sum(acc) / len(acc)
print(band.round(3).to_string())
print(f"\nTotal: {band['profit (M)'].sum():.2f} M EUR; from bands with PD > 15%: "
      f"{band.loc[[b for b in band.index if b.left >= 0.15], 'profit (M)'].sum():.2f} M EUR")
