"""Does it matter whether we actually HAVE a signal on an applicant (signal2 > 0) or not (signal2 == 0)?
(1) how much hidden information the competitors hold in each group, (2) the curse we suffer in each group
under the chosen rule, (3) whether separate margins/thresholds for the two groups improve the rule.
Run from the project folder after run_round1.py:  python analysis/signal_presence.py"""
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
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "output" / "round1"

past = pd.read_csv(HERE / "PastLoans.csv")
new = pd.read_csv(HERE / "NewApplications_Lender2_Round1.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
chosen = strategy["rule"]
seeds = [0, 1, 2]
has = np.where(past["signal2"].to_numpy() > 0, "signal", "no signal")

print(f"New applications: signal2 == 0 for {(new['signal2'] == 0).mean():.1%} of applicants "
      f"({int((new['signal2'] == 0).sum())} people, identifiable by id).")
print(f"Past loans: signal2 == 0 for {(has == 'no signal').mean():.1%}; competitors' signals are zero independently "
      f"(corr of zero indicators ~0.03), so on a no-signal applicant each competitor still has a signal 79% of the time.\n")

# ---------------------------------------------------------------- 1. hidden information by group
print("=== 1. Hidden information std(p_full - p_ours), by income segment and signal presence ===")
U = pd.Series(pF - p2)
print(U.groupby([seg, has]).std().unstack().round(4).to_string())

# ---------------------------------------------------------------- 2. the curse by group under the chosen rule
print("\n=== 2. Chosen rule, base scenario: outcome by signal presence ===")
comp = competitor_rates(p1, p3, seg)
rule = Rule(p2, seg, seg)
r2 = rule.rates(chosen)
r1, r3 = comp["base / base"]
acc = []
for seed in seeds:
    res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
    won = choice == US
    df = pd.DataFrame({"group": has, "won": won, "p2": p2, "y": y, "net": r2 == chosen["tail_rate"],
                       "profit": np.where(won, np.where(y == 1, -LOAN, LOAN * r2), 0.0)})
    g, w = df.groupby("group"), df[won].groupby("group")
    acc.append(pd.DataFrame({"applicants": g.size(), "offered the net": g["net"].mean(), "share won": g["won"].mean(),
                             "predicted PD on won": w["p2"].mean(), "realized default on won": w["y"].mean(),
                             "profit (M)": g["profit"].sum() / 1e6}))
print((sum(acc) / len(acc)).round(3).to_string())

# ---------------------------------------------------------------- 3. separate parameters for the two groups
print("\n=== 3. Re-optimizing with separate margins AND thresholds for signal / no-signal applicants ===")
ev = FastEvaluator(comp, n, seeds, pF)
seg2 = np.char.add(np.char.add(seg.astype(str), " | "), has)
rule2 = Rule(p2, seg2, seg2)
init = {"margins": {k: chosen["margins"][k.split(" | ")[0]] for k in rule2.key_masks},
        "tau": {k: chosen["tau"][k.split(" | ")[0]] for k in rule2.seg_masks},
        "tail_rate": chosen["tail_rate"]}
prm2, best2 = optimize(rule2, ev, init, fixed_tail=chosen["tail_rate"], verbose=False)
avg1, worst1, _ = ev.objective(rule.rates(chosen))
avg2, worst2, _ = ev.objective(rule2.rates(prm2))
print(f"current rule (segment only):        mean {avg1 / 1e6:.3f} M, worst {worst1 / 1e6:.3f} M")
print(f"segment x signal-presence rule:     mean {avg2 / 1e6:.3f} M, worst {worst2 / 1e6:.3f} M")
tab = pd.DataFrame({"margin": prm2["margins"], "threshold": prm2["tau"]})
tab.index = pd.MultiIndex.from_tuples([tuple(k.split(" | ")) for k in tab.index], names=["segment", "group"])
print(tab.unstack("group").round(3).to_string())
