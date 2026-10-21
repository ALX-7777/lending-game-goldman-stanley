"""Refine the 'compromise' schedule found by the Monte Carlo: model + M on the least risky part of the band
(PD up to a cut-off), ramp 0.65 -> 0.828 above.  Same random-rival generator as monte_carlo_competitors.py,
two mixtures, a grid of (M, cut-off).
Run from the project folder:  python analysis/monte_carlo_compromise.py [n_draws]
Output: output/round1/montecarlo_compromise.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import LOAN, PREF, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
REGIMES = ["cheap", "margin", "reject", "net", "ramp"]
MIXTURES = {"baseline (45/15/15/15/10)": [0.45, 0.15, 0.15, 0.15, 0.10],
            "naive rivals (65/10/15/7/3)": [0.65, 0.10, 0.15, 0.07, 0.03],
            "sophisticated rivals (25/15/10/30/20)": [0.25, 0.15, 0.10, 0.30, 0.20]}


def draw_rival(rng, W, p):
    q = np.clip(p, 1e-4, 1 - 1e-4)
    sd = rng.uniform(0, 0.3)
    q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + sd * rng.standard_normal(n))))
    m_safe = rng.uniform(0.01, 0.05)
    slide = rng.random() < 0.10
    safe = np.minimum((q + m_safe) if slide else (be(q) + m_safe), 1.0)
    regime = REGIMES[rng.choice(5, p=W)]
    if regime == "cheap":
        t = rng.choice([np.inf, 0.20, 0.25, 0.30, 0.35])
        return np.where(q <= t, safe, np.inf)
    if regime == "margin":
        return np.where(q <= tau_arr, safe, np.minimum(be(q) + rng.uniform(0.08, 0.40), 1.0))
    if regime == "reject":
        if rng.random() < 0.7:
            return np.where(q <= rng.uniform(0.10, 0.30), safe, np.inf)
        c = rng.uniform(0.25, 0.50)
        return np.where(safe <= c, safe, np.inf)
    if regime == "net":
        thr = rng.choice(["ours", 0.15, 0.20, 0.25])
        tt = tau_arr if thr == "ours" else float(thr)
        X = rng.choice([1.0, 0.99, 0.95, 0.90, 0.85, 0.80]) if rng.random() < 0.5 else rng.uniform(0.60, 0.99)
        return np.where(q <= tt, safe, X)
    thr = rng.choice(["ours", 0.15, 0.20])
    tt = tau_arr if thr == "ours" else float(thr)
    L = rng.uniform(0.50, 0.80)
    H = rng.uniform(max(L + 0.05, 0.80), 0.99)
    frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
    return np.where(q <= tt, safe, L + (H - L) * frac)


def evaluate(r1, r2s, r3, types):
    out = {}
    eff1 = r1 - PREF * (types == 0)
    eff3 = r3 - PREF * (types == 2)
    for a, r2 in r2s.items():
        eff2 = np.where(np.isfinite(r2), r2 - PREF * (types == 1), np.inf)
        E = np.column_stack([eff1, eff2, eff3])
        best = E.min(axis=1)
        choice = np.where(np.isfinite(best), E.argmin(axis=1), -1)
        out[a] = [(LOAN * ((1 - pF[choice == k]) * rk[choice == k] - pF[choice == k])).sum() / 1e6 for k, rk in enumerate([r1, r2, r3])]
    return out


r65 = 0.65 + (0.828 - 0.65) * ramp_frac(p2)
safe = be(p2) + m_arr
OURS = {"ramp 0.65->0.828": np.where(p2 > tau_arr, r65, safe), "model + 0.20 (whole band)": np.where(p2 > tau_arr, np.minimum(be(p2) + 0.20, 1.0), safe)}
for M in [0.15, 0.20, 0.25, 0.30]:
    for cut in [0.15, 0.18, 0.22, 0.25]:
        OURS[f"compromise: model + {M:.2f} if PD <= {cut:.2f}, else ramp"] = np.where(p2 > tau_arr, np.where(p2 <= cut, be(p2) + M, r65), safe)

records = []
for mname, W in MIXTURES.items():
    rng = np.random.default_rng(7)
    print(f"{mname}: {N_DRAWS} draws...", flush=True)
    for d in range(N_DRAWS):
        r1 = draw_rival(rng, W, p1)
        r3 = draw_rival(rng, W, p3)
        types = draw_types(n, int(rng.integers(1 << 30)))
        for a, (q1, q2, q3) in evaluate(r1, OURS, r3, types).items():
            records.append({"mixture": mname, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["mixture", "action"])
S = pd.DataFrame({"mean": g["profit"].mean(), "P5": g["profit"].quantile(0.05), "median": g["profit"].median(),
                  "P95": g["profit"].quantile(0.95), "P(profit < 5)": g["profit"].apply(lambda s: (s < 5).mean()),
                  "P(rank 1)": g["rank"].apply(lambda s: (s == 1).mean()), "P(rank 3)": g["rank"].apply(lambda s: (s == 3).mean())})
S.to_csv(bpm.OUT / "montecarlo_compromise.csv")
for mname in MIXTURES:
    print(f"\n=== {mname} ===")
    print(S.loc[mname].round(2).sort_values("mean", ascending=False).to_string())
