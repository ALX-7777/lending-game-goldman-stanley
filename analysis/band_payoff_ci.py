"""Uncertainty around the band-price payoff matrix.

Two different kinds of 'std':
 (1) WITHIN a world -- sampling noise: which applicants show up in the new 100,000, their types, and who defaults.
     The realized profit on a fresh sample of N applicants is a sum of N iid draws, so its std is sqrt(N) times the
     std of the per-applicant profit measured on the past loans (types drawn at random, realized defaults).
     We report a 95% interval per (action, world) and, more usefully, for the DIFFERENCE between an action and the
     flat net 0.828 in the same world (same applicants -> much tighter).
 (2) ACROSS worlds -- strategic risk: under a prior over competitor behaviours, the distribution of the payoff of
     each action over worlds (mean, std, 5th/95th percentiles, probability of a disappointing round).
Run from the project folder after band_payoff_matrix.py:  python analysis/band_payoff_ci.py
Outputs: output/round1/band_matrix_ci.csv, band_matrix_risk.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402  (module-level definitions only; its main() does not run)
from lending_game.simulator import LOAN, PREF, US, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
n, y, pF, seeds = bpm.n, bpm.y, bpm.pF, bpm.seeds
CANDIDATES = ["flat net 0.828", "flat net 0.700", "ramp 0.60->0.828", "ramp 0.65->0.828", "ramp 0.70->0.828",
              "ramp 0.60->0.900", "model + 0.15", "model + 0.20", "hybrid: model + 0.15 to PD 0.15, then net 0.828",
              "no offer in band"]
REF = "flat net 0.828"
Z = 1.96

# ---------------------------------------------------------------- (1) within-world CIs
records = []
for wname, (r1, r3) in bpm.cases.items():
    per_action = {a: {"mean": [], "var": [], "mean_exp": [], "dmean": [], "dvar": []} for a in CANDIDATES}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        types = draw_types(n, seed)
        eff1 = r1 - PREF * (types == 0) + rng.uniform(0, 1e-9, n)
        eff3 = r3 - PREF * (types == 2) + rng.uniform(0, 1e-9, n)
        comp_min = np.minimum(eff1, eff3)
        our_fan = types == US
        prof_ref = None
        profs = {}
        for a in CANDIDATES:
            r2 = bpm.ACTIONS[a]
            eff2 = np.where(np.isfinite(r2), r2 - PREF * our_fan, np.inf)
            won = eff2 < comp_min
            r_won = np.where(won, r2, 0.0)
            prof = np.where(won, np.where(y == 1, -LOAN, LOAN * r_won), 0.0)         # realized, per applicant
            prof_exp = np.where(won, LOAN * ((1 - pF) * r_won - pF), 0.0)             # expected, per applicant
            profs[a] = prof
            per_action[a]["mean"].append(prof.sum())
            per_action[a]["var"].append(n * prof.var(ddof=1))                        # var of the total on a fresh sample
            per_action[a]["mean_exp"].append(prof_exp.sum())
        for a in CANDIDATES:
            d = profs[a] - profs[REF]
            per_action[a]["dmean"].append(d.sum())
            per_action[a]["dvar"].append(n * d.var(ddof=1))
    for a in CANDIDATES:
        s = per_action[a]
        mean, std = np.mean(s["mean"]) / 1e6, np.sqrt(np.mean(s["var"])) / 1e6
        dmean, dstd = np.mean(s["dmean"]) / 1e6, np.sqrt(np.mean(s["dvar"])) / 1e6
        records.append({"world": wname, "regime": bpm.regime_of[wname], "action": a,
                        "expected (M)": np.mean(s["mean_exp"]) / 1e6, "realized mean (M)": mean, "std (M)": std,
                        "ci95 low": mean - Z * std, "ci95 high": mean + Z * std,
                        "diff vs flat 0.828 (M)": dmean, "diff std (M)": dstd,
                        "diff ci95 low": dmean - Z * dstd, "diff ci95 high": dmean + Z * dstd})
CI = pd.DataFrame(records)
CI.to_csv(bpm.OUT / "band_matrix_ci.csv", index=False)

print("=== 1. Within-world 95% intervals for the realized profit on a fresh sample of 100,000 applicants (M EUR) ===")
show = ["flat net 0.828", "ramp 0.60->0.828", "ramp 0.70->0.828", "flat net 0.700", "model + 0.20"]
cell = CI[CI.action.isin(show)].copy()
cell["cell"] = cell.apply(lambda r: f"{r['realized mean (M)']:.1f} +/- {Z * r['std (M)']:.1f}", axis=1)
print(cell.pivot(index="world", columns="action", values="cell")[show].loc[list(bpm.WORLDS)].to_string())
print("\ntypical half-width of the 95% interval (M EUR), by regime and action:")
hw = CI.assign(hw=Z * CI["std (M)"]).pivot_table(index="regime", columns="action", values="hw", aggfunc="mean")
print(hw[show].round(2).to_string())

print("\n=== 2. Difference vs the flat net 0.828, same world, same applicants: mean and 95% interval (M EUR) ===")
diff = CI[CI.action.isin(["ramp 0.60->0.828", "ramp 0.70->0.828", "flat net 0.700", "model + 0.20"])].copy()
diff["cell"] = diff.apply(lambda r: f"{r['diff vs flat 0.828 (M)']:+.2f} [{r['diff ci95 low']:+.2f}, {r['diff ci95 high']:+.2f}]"
                          + (" *" if r["diff ci95 low"] > 0 or r["diff ci95 high"] < 0 else ""), axis=1)
print(diff.pivot(index="world", columns="action", values="cell").loc[list(bpm.WORLDS)].to_string())
print("(* = the 95% interval of the difference excludes zero)")

# ---------------------------------------------------------------- (2) across-world risk under priors
print("\n=== 3. Across-world risk under each prior: distribution of the expected payoff over worlds (M EUR) ===")
E = CI.pivot(index="world", columns="action", values="expected (M)").loc[list(bpm.WORLDS)]
rows = []
for pname, prior in bpm.PRIORS.items():
    w = bpm.world_weights(prior).loc[E.index].to_numpy()
    for a in CANDIDATES:
        v = E[a].to_numpy()
        order = np.argsort(v)
        cum = np.cumsum(w[order])
        q = lambda p: v[order][np.searchsorted(cum, p)]
        mean = float(np.dot(w, v))
        std = float(np.sqrt(np.dot(w, (v - mean) ** 2)))
        rows.append({"prior": pname, "action": a, "mean": mean, "std": std, "P5": q(0.05), "median": q(0.5), "P95": q(0.95),
                     "P(payoff < 3)": float(w[v < 3].sum()), "P(payoff < 5)": float(w[v < 5].sum()),
                     "P(payoff > 20)": float(w[v > 20].sum())})
RISK = pd.DataFrame(rows)
RISK.to_csv(bpm.OUT / "band_matrix_risk.csv", index=False)
for pname in bpm.PRIORS:
    print(f"\n--- prior: {pname} ---")
    print(RISK[RISK.prior == pname].drop(columns="prior").set_index("action").round(2).to_string())
