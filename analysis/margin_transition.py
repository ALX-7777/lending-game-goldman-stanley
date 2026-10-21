"""Smoothing the margin transition at the threshold.

Unified rule:  rate(p) = adjusted break-even(p) + margin(p), with margin rising from the competitive level on the safe part
to X = 0.05 in the band as w(p) = sigmoid((p - (tau + delta)) / width) goes from 0 to 1.  Operationally:
    rate = r_safe + w(p) * (min(adjusted break-even + X, 0.989) - r_safe),   r_safe = p/(1-p) + m(segment).
Candidates = (delta, width) pairs; evaluated with exact types and fair ties against the seven rival profiles.
Run from the project folder:  python analysis/margin_transition.py [n_draws]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3, seg = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CAP, X = 0.989, 0.05
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
safe = be(p2) + m_arr
band = np.minimum(adj + X, CAP)


def rule(delta, width, p=p2, tau=tau_arr, safe_=None, band_=None):
    safe_ = safe if safe_ is None else safe_
    band_ = band if band_ is None else band_
    w = 1 / (1 + np.exp(-(p - (tau + delta)) / width))
    return np.round(safe_ + w * (band_ - safe_), 4)


TRANS = {"current (1-point step at tau)": (0.005, 0.0017), "delta 1pt, width 1pt (~8-14%)": (0.01, 0.01),
         "delta 2pt, width 1.5pt (~8-16%)": (0.02, 0.015), "delta 2pt, width 2pt (~7-18%)": (0.02, 0.02),
         "delta 3pt, width 2.5pt (~7-20%)": (0.03, 0.025)}
CANDS = {k: rule(d, w) for k, (d, w) in TRANS.items()}

# what each candidate charges in the 5-10k segment (tau 10%, margin 3.5%) at a few PD levels
grid = np.array([0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.20, 0.25])
adj_g = np.interp(grid, floor["pd"], floor["adjusted_breakeven"])
show = pd.DataFrame({"PD": grid * 100, "plain break-even": be(grid) * 100, "adjusted break-even": adj_g * 100})
for k, (d, w) in TRANS.items():
    show[k] = rule(d, w, p=grid, tau=np.full(len(grid), 0.10), safe_=be(grid) + 0.035, band_=np.minimum(adj_g + X, CAP)) * 100
print("=== Rate (%) by PD in the 5-10k segment for each transition ===")
print(show.round(1).to_string(index=False))


def evaluate_exact(r1, r2s, r3):
    out = {}
    for a, r2 in r2s.items():
        profits = np.zeros(3)
        for t in range(3):
            E = np.column_stack([r1 - PREF * (t == 0), np.where(np.isfinite(r2), r2 - PREF * (t == 1), np.inf), r3 - PREF * (t == 2)])
            m = E.min(axis=1)
            tie = (E <= m[:, None] + 1e-12) & np.isfinite(m)[:, None]
            n_min = tie.sum(axis=1)
            for k, rk in enumerate([r1, r2, r3]):
                wp = np.where(tie[:, k], 1.0 / np.maximum(n_min, 1), 0.0) / 3
                profits[k] += (wp * LOAN * ((1 - pF) * np.where(np.isfinite(rk), rk, 0.0) - pF)).sum() / 1e6
        out[a] = profits
    return out


PROFILES = {"likeus": 0.40, "smart": 0.20, "clones": 0.10, "verysmart": 0.10, "mixed": 0.10, "netters": 0.05, "modelfollowers": 0.05}
records = []
for profile, wt in PROFILES.items():
    W = list(rivals.MIXTURES[profile].values())[0]
    rng = np.random.default_rng(6060)
    print(f"{profile}: {N_DRAWS} draws...", flush=True)
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        for a, (q1, q2, q3) in evaluate_exact(r1, CANDS, r3).items():
            records.append({"profile": profile, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["action", "profile"])
wts = np.array(list(PROFILES.values()))
order = list(PROFILES)
M = g["profit"].mean().unstack("profile")[order].loc[list(CANDS)]
R1 = g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile")[order].loc[list(CANDS)]
R3 = g["rank"].apply(lambda s: (s == 3).mean()).unstack("profile")[order].loc[list(CANDS)]
S = pd.DataFrame({"mean | prior": M.to_numpy() @ wts, "P(first) | prior": R1.to_numpy() @ wts, "P(last) | prior": R3.to_numpy() @ wts,
                  "mean | likeus": M["likeus"], "mean | smart": M["smart"], "mean | mixed": M["mixed"]}, index=list(CANDS))
print("\n=== Transition width: prior-weighted results (exact types, fair ties) ===")
print(S.round(3).to_string())
S.to_csv(bpm.OUT / "margin_transition.csv")
