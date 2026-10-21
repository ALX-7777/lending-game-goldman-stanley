"""Robustness of the round-1 rule: how much of the profit comes from the 98.9% safety net,
and does the low rejection threshold survive a different weighting of competitor scenarios?
Run from the project folder after run_round1.py:  python analysis/robustness_round1.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lending_game.competitors import competitor_rates  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import LOAN, US, FastEvaluator, draw_types, simulate  # noqa: E402
from lending_game.strategy import Rule, optimize  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
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
seeds = [0, 1, 2]

comp_all = competitor_rates(p1, p3, seg)
ev_all = FastEvaluator(comp_all, n, seeds, pF)
rule = Rule(p2, seg, seg)


def table(params, label):
    avg, worst, per = ev_all.objective(rule.rates(params))
    r2 = rule.rates(params)
    # share and profit split in the base scenario (seed 0)
    r1, r3 = comp_all["base / base"]
    types = draw_types(n, 0)
    res, choice = simulate(np.column_stack([r1, r2, r3]), types, y, pF, 0)
    won = choice == US
    at_net = won & (r2 == params["tail_rate"]) if params["tail_rate"] else np.zeros(n, bool)
    prof = np.where(y == 1, -LOAN, LOAN * np.where(np.isfinite(r2), r2, 0))
    return pd.Series({"mean over scenarios (M)": avg / 1e6, "worst scenario (M)": worst / 1e6,
                      "base/base (M)": per["base / base"] / 1e6,
                      "base: share": won.mean(), "base: loans at net": int(at_net.sum()),
                      "base: profit from net loans (M)": prof[at_net].sum() / 1e6,
                      "base: profit from normal loans (M)": prof[won & ~at_net].sum() / 1e6,
                      "applicants offered the net": float((r2 == params["tail_rate"]).mean()) if params["tail_rate"] else 0.0},
                     name=label)


rows = [table(chosen, "optimized (tau as chosen)")]
for t in [0.15, 0.20, 0.25, 0.30]:
    prm = copy.deepcopy(chosen)
    prm["tau"] = {s: t for s in prm["tau"]}
    rows.append(table(prm, f"same margins, tau={t:.2f} everywhere"))
prm = copy.deepcopy(chosen)
prm["tail_rate"] = None
rows.append(table(prm, "optimized but NO safety net (no offer instead)"))
print("=== Sensitivity to the safety-net threshold (tau) ===")
print(pd.DataFrame(rows).round(3).to_string())

print("\n=== Re-optimizing under different scenario weightings ===")
variants = {
    "all 12 scenarios (as in run)": comp_all,
    "excluding cautious/cautious": {k: v for k, v in comp_all.items() if k != "cautious / cautious"},
    "only 'no rejection' competitors": {k: v for k, v in comp_all.items() if "no_reject" in k or "norej" in k or "aggressive" in k},
    "only rejecting competitors": {k: v for k, v in comp_all.items() if not ("no_reject" in k or "norej" in k or "aggressive" in k)},
}
summary = []
for label, comp in variants.items():
    ev = FastEvaluator(comp, n, seeds, pF)
    prm, best = optimize(rule, ev, copy.deepcopy(chosen), verbose=False)
    avg_all, worst_all, _ = ev_all.objective(rule.rates(prm))
    summary.append(pd.Series({"tau": json.dumps(prm["tau"]), "margins": json.dumps(prm["margins"]),
                              "tail": prm["tail_rate"], "own objective (M)": best / 1e6,
                              "mean over ALL scenarios (M)": avg_all / 1e6, "worst over ALL (M)": worst_all / 1e6}, name=label))
print(pd.DataFrame(summary).round(3).to_string())
