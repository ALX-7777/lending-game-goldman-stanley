"""What if another team uses a safety net like ours?

'Clone' competitors apply OUR rule (margins and low thresholds by income segment) to their own PD
estimate, with a net of their choosing. Unlike the 'base' competitors (who only net above a 25% PD),
clones abandon the whole 10-25% PD band to the net, so that band becomes a price war at net level.
We ask: (1) how does each candidate net of ours fare against clones at various nets (including an
exact tie at 0.898)?  (2) does re-optimizing the full rule with clones in the mix change anything?
Run from the project folder after run_round1.py:  python analysis/clone_scenarios.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lending_game.strategy as strategy_mod  # noqa: E402
from lending_game.competitors import ARCHETYPES, clone_rates, competitor_rates, price  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import FastEvaluator  # noqa: E402
from lending_game.strategy import Rule, optimize  # noqa: E402

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


def clone(p, net, scale=1.0):
    return clone_rates(p, seg, chosen, net, scale)


base, norej = ARCHETYPES["base"], ARCHETYPES["no_reject"]
rb1, rb3 = price(p1, seg=seg, **base), price(p3, seg=seg, **base)

# ------------------------------------------------------------------ 1. the contested band
tau = np.array([chosen["tau"][s] for s in seg])
we_net, c1_net, c3_net = p2 > tau, p1 > tau, p3 > tau
all3 = we_net & c1_net & c3_net
print("=== 1. Who nets whom (clones with our thresholds) ===")
print(f"we net {we_net.mean():.1%} of applicants; both clones also net {all3.mean():.1%} of applicants "
      f"(= {all3.sum() / we_net.sum():.0%} of those we net)")
d = y[all3].mean()
print(f"default rate in the band where all three net: {d:.1%}  ->  break-even net rate {d / (1 - d):.2f}")
some = we_net & ~all3
print(f"where we net but at least one clone lends at a normal rate ({some.mean():.1%} of applicants), "
      f"default rate {y[some].mean():.1%}: the clone wins those, we get nothing\n")

# ------------------------------------------------------------------ 2. our net vs theirs
cases = {"nobody nets: base/base": (rb1, rb3),
         "nobody nets: no_reject/no_reject": (price(p1, seg=seg, **norej), price(p3, seg=seg, **norej))}
for X in [0.99, 0.95, 0.90, 0.85]:
    cases[f"base(25%) both net {X:.2f}"] = (price(p1, seg=seg, net=X, **base), price(p3, seg=seg, net=X, **base))
CLONE_NETS = [0.989, 0.95, 0.92, 0.90, 0.898, 0.89, 0.87, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
for X in CLONE_NETS:
    cases[f"both clones net {X:.3f}"] = (clone(p1, X), clone(p3, X))
for X in [0.95, 0.898, 0.85, 0.75]:
    cases[f"clone {X:.3f} / base"] = (clone(p1, X), rb3)
for X in [0.95, 0.898, 0.85]:
    cases[f"moderate clones (tau x2) net {X:.3f}"] = (clone(p1, X, 2.0), clone(p3, X, 2.0))

ev = FastEvaluator(cases, n, seeds, pF)
OUR_NETS = [0.989, 0.95, 0.92, 0.898, 0.877, 0.86, 0.85, 0.83, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, None]


def label(net):
    return "no offer" if net is None else f"{net:.3f}"


def profit_table(rule_obj, params):
    rows = {}
    for net in OUR_NETS:
        res = ev.profits(rule_obj.rates({**params, "tail_rate": net}))
        rows[label(net)] = {k: np.mean(v) / 1e6 for k, v in res.items()}
    return pd.DataFrame(rows).T


T = profit_table(rule, chosen)
groups = {
    "nobody nets": [c for c in T.columns if c.startswith("nobody")],
    "both clones": [c for c in T.columns if c.startswith("both clones")],
    "one clone / base": [c for c in T.columns if c.startswith("clone")],
    "other netters": [c for c in T.columns if c.startswith("base(") or c.startswith("moderate")],
}
print("=== 2. Our expected profit (M EUR) by our net rate (rows) against each competitor case (columns) ===")
for g, cols in groups.items():
    print(f"\n--- {g} ---")
    print(T[cols].rename(columns=lambda c: c.replace("both clones net ", "clones ").replace("nobody nets: ", "")
                         .replace("moderate clones (tau x2) net ", "moderate ")).round(2).to_string())

print("\n=== 3. Best response per competitor case, and what 0.898 leaves on the table ===")
br = pd.DataFrame({"best net for us": T.idxmax(axis=0), "best profit": T.max(axis=0),
                   "profit at 0.898": T.loc["0.898"], "regret of 0.898": T.max(axis=0) - T.loc["0.898"]})
print(br.round(2).to_string())

print("\n=== 4. Summary by world (mean M EUR), plus the worst regret across all cases ===")
worlds = {
    "nobody nets": groups["nobody nets"],
    "nets >= 0.95": [c for c in T.columns if any(f"{x:.3f}" in c or f"{x:.2f}" in c for x in [0.989, 0.99, 0.95])],
    "nets 0.87-0.92": [c for c in T.columns if any(f"{x:.3f}" in c or f"{x:.2f}" in c for x in [0.92, 0.90, 0.898, 0.89, 0.87])],
    "nets <= 0.85": [c for c in T.columns if any(f"{x:.3f}" in c or f"{x:.2f}" in c for x in [0.85, 0.80, 0.75, 0.70, 0.60, 0.50])],
}
S = pd.DataFrame({w: T[cols].mean(axis=1) for w, cols in worlds.items()})
S["all cases"] = T.mean(axis=1)
S["max regret"] = (T.max(axis=0) - T).max(axis=1)
print(S.round(2).to_string())

# ------------------------------------------------------------------ 5. re-optimize with clones in the mix
print("\n=== 5. Re-optimizing margins, thresholds and net with clone scenarios added ===")
strategy_mod.TAIL_GRID = [None, 0.5, 0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.86, 0.877, 0.898, 0.92, 0.95, 0.989]
comp_ext = competitor_rates(p1, p3, seg)
for name in ["both clones net 0.950", "both clones net 0.898", "both clones net 0.850", "both clones net 0.750",
             "clone 0.898 / base", "base(25%) both net 0.95"]:
    comp_ext[name] = cases[name]
ev_ext = FastEvaluator(comp_ext, n, seeds, pF)
prm_ext, best_ext = optimize(rule, ev_ext, copy.deepcopy(chosen), verbose=False)
print("re-optimized rule:", json.dumps(prm_ext))
rows = []
for lab, prm in [("current rule (net 0.898)", chosen), ("re-optimized with clones", prm_ext)]:
    res = {k: np.mean(v) / 1e6 for k, v in ev.profits(rule.rates(prm)).items()}
    r = pd.Series(res)
    rows.append(pd.Series({w: r[cols].mean() for w, cols in worlds.items()} | {"all cases": r.mean(), "min": r.min()},
                          name=lab))
print(pd.DataFrame(rows).round(2).to_string())

# ------------------------------------------------------------------ 6. decision table under explicit priors
print("\n=== 6. Prior-weighted expected profit (M EUR) by our net: current thresholds vs clone-aware thresholds ===")
T2 = profit_table(rule, prm_ext)  # clone-aware margins/thresholds, our net varied along the rows
T.to_csv(OUT / "clone_profit_table_current_rule.csv")
T2.to_csv(OUT / "clone_profit_table_clone_aware_rule.csv")
families = {
    "nobody nets": [c for c in T.columns if c.startswith("nobody")],
    "base nets (above 25%)": [c for c in T.columns if c.startswith("base(")],
    "moderate clones": [c for c in T.columns if c.startswith("moderate")],
    "one clone / base": [c for c in T.columns if c.startswith("clone")],
    "both clones": [c for c in T.columns if c.startswith("both clones")],
}
priors = {  # probability of each family; uniform over the cases inside a family
    "skeptical": [0.70, 0.15, 0.05, 0.07, 0.03],
    "moderate":  [0.50, 0.20, 0.10, 0.12, 0.08],
    "paranoid":  [0.30, 0.20, 0.15, 0.15, 0.20],
}
print("family probabilities:", {k: dict(zip(families, v)) for k, v in priors.items()})
out = {}
for pname, probs in priors.items():
    for lab, tab in [("current thresholds", T), ("clone-aware thresholds", T2)]:
        col = sum(pr * tab[cols].mean(axis=1) for pr, (fam, cols) in zip(probs, families.items()))
        out[f"{pname} / {lab}"] = col
D = pd.DataFrame(out)
print(D.round(2).to_string())
print("\nbest net per column:\n" + D.idxmax().to_string())
