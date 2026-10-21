"""Threshold + net, or model rate + an increasing safety margin?  And where is OUR signal worth most?

Rule families compared on the same simulator (12 competitor scenarios, 3 type draws, expected profit):
  (a) chosen rule:   r = be(p) + m(seg)                 if p <= tau(seg), else net 0.828
  (b) smooth, no net: r = be(p) + m(seg) + b * (p - 0.05)+   for everyone (no threshold, no net)
  (c) smooth + net:   r = be(p) + m(seg) + b * (p - 0.05)+   if p <= tau(seg), else net 0.828
where be(p) = p/(1-p) and b is a slope: the margin grows with the predicted PD.
Run from the project folder after run_round1.py:  python analysis/margin_vs_threshold.py"""
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
from lending_game.strategy import MARGIN_GRID, TAU_GRID  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "output" / "round1"

past = pd.read_csv(HERE / "PastLoans.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
pX, p1, p2, p3, pF = z["pX"], z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
chosen = strategy["rule"]
seeds = [0, 1, 2]
comp = competitor_rates(p1, p3, seg)
ev = FastEvaluator(comp, n, seeds, pF)
SLOPE_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


class SlopeRule:
    def __init__(self):
        self.seg_masks = {s: seg == s for s in sorted(set(seg))}

    def rates(self, prm):
        m = np.zeros(n)
        tau = np.zeros(n)
        for s, mask in self.seg_masks.items():
            m[mask] = prm["margins"][s]
            tau[mask] = prm["tau"][s]
        r = np.minimum(p2 / (1 - p2) + m + prm["slope"] * np.maximum(p2 - 0.05, 0), 1.0)
        tail = prm["tail_rate"]
        return np.where(p2 > tau, np.inf if tail is None else tail, r)


def optimize_slope(rule, init, search_tau, max_sweeps=8, min_gain=1_000.0):
    prm = copy.deepcopy(init)
    best = ev.objective(rule.rates(prm))[0]
    for _ in range(max_sweeps):
        improved = False
        for s in rule.seg_masks:
            for mval in MARGIN_GRID:
                t = copy.deepcopy(prm); t["margins"][s] = mval
                v = ev.objective(rule.rates(t))[0]
                if v > best + min_gain: best, prm, improved = v, t, True
        for b in SLOPE_GRID:
            t = copy.deepcopy(prm); t["slope"] = b
            v = ev.objective(rule.rates(t))[0]
            if v > best + min_gain: best, prm, improved = v, t, True
        if search_tau:
            for s in rule.seg_masks:
                for tv in TAU_GRID:
                    t = copy.deepcopy(prm); t["tau"][s] = tv
                    v = ev.objective(rule.rates(t))[0]
                    if v > best + min_gain: best, prm, improved = v, t, True
        if not improved:
            break
    return prm, best


# ---------------------------------------------------------------- 1. where is our signal worth most?
print("=== 1. How much our signal moves the PD, by income segment (std of p_ours - p_X-only), vs what competitors hide ===")
print(pd.DataFrame({"our signal: std(p2 - pX)": pd.Series(p2 - pX).groupby(seg).std(),
                    "competitors' hidden info: std(pF - p2)": pd.Series(pF - p2).groupby(seg).std()}).round(4).to_string())

# ---------------------------------------------------------------- 2. the 'signal says safe' group at low income
print("\n=== 2. Low-income applicants our signal flags as safe (p2 <= threshold) vs. not, chosen rule, base scenario ===")
rule0 = SlopeRule()
prm0 = {**copy.deepcopy(chosen), "slope": 0.0}
r2 = rule0.rates(prm0)
r1, r3 = comp["base / base"]
tau_arr = np.array([chosen["tau"][s] for s in seg])
low = np.isin(seg, ["0", "1-5k", "5-10k"])
group = np.where(~low, "income > 10k", np.where(p2 <= tau_arr, "low income, signal says safe", "low income, risky -> net"))
acc = []
for seed in seeds:
    res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
    won = choice == US
    df = pd.DataFrame({"group": group, "won": won, "p2": p2, "pX": pX, "y": y, "rate": r2,
                       "best competitor rate": np.minimum(r1, r3),
                       "profit": np.where(won, np.where(y == 1, -LOAN, LOAN * r2), 0.0)})
    g, w = df.groupby("group"), df[won].groupby("group")
    acc.append(pd.DataFrame({"applicants": g.size(), "PD without signal (pX)": g["pX"].mean(), "our PD (p2)": g["p2"].mean(),
                             "our rate": g["rate"].mean(), "best competitor rate": g["best competitor rate"].mean(),
                             "share won": g["won"].mean(), "predicted PD on won": w["p2"].mean(),
                             "realized default on won": w["y"].mean(), "profit (M)": g["profit"].sum() / 1e6}))
print((sum(acc) / len(acc)).round(3).to_string())

# ---------------------------------------------------------------- 3. rule families
print("\n=== 3. Optimized rule families (expected profit, M EUR) ===")
rows = []
avg, worst, _ = ev.objective(r2)
rows.append(pd.Series({"mean": avg / 1e6, "worst": worst / 1e6, "slope b": 0.0, "net": chosen["tail_rate"],
                       "thresholds": json.dumps(chosen["tau"])}, name="(a) chosen: segment margin + threshold + net"))

init_b = {"margins": copy.deepcopy(chosen["margins"]), "tau": {s: 1.0 for s in chosen["tau"]}, "slope": 0.5, "tail_rate": None}
prm_b, best_b = optimize_slope(rule0, init_b, search_tau=False)
avg, worst, _ = ev.objective(rule0.rates(prm_b))
rows.append(pd.Series({"mean": avg / 1e6, "worst": worst / 1e6, "slope b": prm_b["slope"], "net": None,
                       "thresholds": "none (lend to everyone)"}, name="(b) model rate + increasing margin, no net"))

init_c = {**copy.deepcopy(chosen), "slope": 0.0}
prm_c, best_c = optimize_slope(rule0, init_c, search_tau=True)
avg, worst, _ = ev.objective(rule0.rates(prm_c))
rows.append(pd.Series({"mean": avg / 1e6, "worst": worst / 1e6, "slope b": prm_c["slope"], "net": prm_c["tail_rate"],
                       "thresholds": json.dumps(prm_c["tau"])}, name="(c) increasing margin + threshold + net"))
print(pd.DataFrame(rows).round(3).to_string())
print("\n(b) margins by segment:", json.dumps(prm_b["margins"]))
print("(c) margins by segment:", json.dumps(prm_c["margins"]))

# what (b) charges and wins, by predicted-PD band, base scenario
print("\n=== 4. Family (b) in the base scenario, by band of predicted PD ===")
rb = rule0.rates(prm_b)
acc = []
for seed in seeds:
    res, choice = simulate(np.column_stack([r1, rb, r3]), draw_types(n, seed), y, pF, seed)
    won = choice == US
    df = pd.DataFrame({"band": pd.cut(p2, [0, .05, .10, .15, .20, .25, .30, .40, 1.0]), "won": won, "p2": p2, "y": y,
                       "rate": rb, "profit": np.where(won, np.where(y == 1, -LOAN, LOAN * rb), 0.0)})
    g, w = df.groupby("band", observed=True), df[won].groupby("band", observed=True)
    acc.append(pd.DataFrame({"applicants": g.size(), "our rate": g["rate"].mean(), "share won": g["won"].mean(),
                             "predicted PD on won": w["p2"].mean(), "realized default on won": w["y"].mean(),
                             "profit (M)": g["profit"].sum() / 1e6}))
print((sum(acc) / len(acc)).round(3).to_string())
