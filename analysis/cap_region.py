"""Is the 0.989 cap region (applicants whose adjusted break-even + X exceeds the cap) worth keeping?
Options: keep the cap; no offer in the cap region; cap up to 30% PD and no offer above.
For each rival profile: expected loans won in that region, their expected default rate, the expected profit on them,
and the total expected profit of the three options (exact types, fair ties).
Run from the project folder:  python analysis/cap_region.py [n_draws]"""
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
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
_args = [a for a in sys.argv[1:] if a != "seg"]
N_DRAWS = int(_args[0]) if _args else 60
SEG_FLOOR = "seg" in sys.argv  # python analysis/cap_region.py 60 seg -> income-segment floor
CAP, X = 0.989, 0.05
if SEG_FLOOR:
    fs = pd.read_csv(bpm.OUT / "curse_floor_by_segment.csv")
    adj = np.full(n, np.nan)
    for s_ in fs["segment"].unique():
        sub, mk_ = fs[fs["segment"] == s_], bpm.seg == s_
        adj[mk_] = np.interp(p2[mk_], sub["pd"], sub["adjusted_breakeven"])
    print("income-segment floor")
else:
    floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
    adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
safe = be(p2) + m_arr
band = np.minimum(adj + X, CAP)
w = 1 / (1 + np.exp(-(p2 - tau_arr - 0.01) / 0.01))
current = np.round(safe + w * (band - safe), 4)
cap_region = (adj + X >= CAP) & (p2 > tau_arr)
print(f"cap region: {cap_region.sum()} applicants (PD from {p2[cap_region].min():.3f}), expected default (full info) {pF[cap_region].mean():.3f}")
OPTIONS = {"keep the cap 0.989": current,
           "no offer in the cap region": np.where(cap_region, np.inf, current),
           "cap up to 30% PD, no offer above": np.where(cap_region & (p2 > 0.30), np.inf, current)}


def our_winprob(r1, r2, r3):
    wp = np.zeros(n)
    for t in range(3):
        E = np.column_stack([r1 - PREF * (t == 0), np.where(np.isfinite(r2), r2 - PREF * (t == 1), np.inf), r3 - PREF * (t == 2)])
        m = E.min(axis=1)
        tie = (E <= m[:, None] + 1e-12) & np.isfinite(m)[:, None]
        wp += np.where(tie[:, 1], 1.0 / np.maximum(tie.sum(axis=1), 1), 0.0) / 3
    return wp


PROFILES = {"likeus": 0.40, "smart": 0.20, "clones": 0.10, "verysmart": 0.10, "mixed": 0.10, "netters": 0.05, "modelfollowers": 0.05}
rows = []
for profile in PROFILES:
    W = list(rivals.MIXTURES[profile].values())[0]
    rng = np.random.default_rng(7070)
    acc = {o: {"total": 0.0, "loans": 0.0, "pf": 0.0, "profit": 0.0} for o in OPTIONS}
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        for o, r2 in OPTIONS.items():
            wp = our_winprob(r1, r2, r3)
            r2f = np.where(np.isfinite(r2), r2, 0.0)
            prof_i = wp * LOAN * ((1 - pF) * r2f - pF)
            acc[o]["total"] += prof_i.sum() / 1e6 / N_DRAWS
            acc[o]["loans"] += wp[cap_region].sum() / N_DRAWS
            acc[o]["pf"] += (wp[cap_region] * pF[cap_region]).sum() / N_DRAWS
            acc[o]["profit"] += prof_i[cap_region].sum() / 1e6 / N_DRAWS
    for o, a in acc.items():
        rows.append({"profile": profile, "option": o, "total expected profit (M)": a["total"],
                     "expected loans won in cap region": a["loans"],
                     "their expected default": a["pf"] / a["loans"] if a["loans"] > 0 else np.nan,
                     "profit on them (M)": a["profit"]})
    print(f"{profile} done", flush=True)
T = pd.DataFrame(rows)
print("\n=== Cap region by rival profile ===")
print(T.round(3).to_string(index=False))
wts = pd.Series(PROFILES)
P = T.pivot(index="option", columns="profile", values="total expected profit (M)")[list(PROFILES)]
print("\n=== Prior-weighted total expected profit (M EUR) ===")
print((P @ wts).round(3).to_string())
C = T.pivot(index="option", columns="profile", values="profit on them (M)")[list(PROFILES)]
print("\n=== Prior-weighted profit on the cap-region applicants (M EUR) ===")
print((C @ wts).round(3).to_string())
