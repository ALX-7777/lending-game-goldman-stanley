"""Two checks requested by the advisor.
(1) Does the selected default rate pi~(PD) depend on the common pricing rule?  With the 2% preference it can: the
    2-point rent corresponds to a different PD gap under a steeper or flatter rule.  We recompute pi~ under several
    symmetric rules and report the deviations.
(2) Does pi~ differ by income segment at the same PD?  We compute pi~ by (segment x PD bin).
Run from the project folder:  python analysis/curse_sensitivity.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, y, pF, p1, p2, p3, seg = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
EDGES = np.arange(0.0, 0.405, 0.01)


def winprob(rule):
    """Probability that we win each applicant when all three lenders apply `rule` (a function of own PD) -- exact types."""
    r1, r2, r3 = rule(p1), rule(p2), rule(p3)
    wp = np.zeros(n)
    for t in range(3):
        wp += ((r2 - PREF * (t == 1)) < np.minimum(r1 - PREF * (t == 0), r3 - PREF * (t == 2))) / 3
    return wp


def curve(wp, mask=None):
    mask = np.ones(n, bool) if mask is None else mask
    out = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mk = mask & (p2 > lo) & (p2 <= hi)
        w = wp[mk]
        if mk.sum() >= 30 and w.sum() >= 5:
            out[round((lo + hi) / 2, 3)] = (w * pF[mk]).sum() / w.sum()
    return pd.Series(out)


def final_rule(p):  # our submitted shape, applied symmetrically (adjusted break-even + 0.05, logistic transition, cap)
    adj = np.interp(p, floor["pd"], floor["adjusted_breakeven"])
    safe = be(p) + m_arr
    band = np.minimum(adj + 0.05, 0.989)
    w = 1 / (1 + np.exp(-(p - tau_arr - 0.01) / 0.01))
    return safe + w * (band - safe)


RULES = {"break-even + m(s)  [used for the fit]": lambda p: be(p) + m_arr,
         "final rule applied by everyone (adjusted BE + 0.05)": final_rule,
         "lecture formula p + 3%": lambda p: p + 0.03,
         "twice the break-even + m(s)": lambda p: 2 * be(p) + m_arr,
         "break-even + 10%": lambda p: be(p) + 0.10,
         "break-even + m(s), no 2% preference": None}
curves = {}
for name, rule in RULES.items():
    if rule is None:
        r1, r2, r3 = be(p1) + m_arr, be(p2) + m_arr, be(p3) + m_arr
        wp = (r2 < np.minimum(r1, r3)).astype(float)
    else:
        wp = winprob(rule)
    curves[name] = curve(wp)
C = pd.DataFrame(curves)
ref = C.iloc[:, 0]
print("=== (1) Selected default rate pi~ by PD bin under different common pricing rules (percent) ===")
show = C.loc[[0.065, 0.085, 0.105, 0.125, 0.145, 0.165, 0.185, 0.205, 0.225, 0.245, 0.265]]
print((show * 100).round(1).to_string())
dev = (C.sub(ref, axis=0)).abs()
print("\nmax |deviation| from the fitted curve, by rule (points of default), over PD 5-25%:")
print((dev.loc[(dev.index >= 0.05) & (dev.index <= 0.25)].max() * 100).round(2).to_string())
print("mean |deviation| over PD 5-25%:")
print((dev.loc[(dev.index >= 0.05) & (dev.index <= 0.25)].mean() * 100).round(2).to_string())

# ---------------------------------------------------------------- (2) by income segment
wp0 = winprob(RULES["break-even + m(s)  [used for the fit]"])
by_seg = {s: curve(wp0, seg == s) for s in ["0", "1-5k", "5-10k", "10-20k"]}
S = pd.DataFrame(by_seg)
S["all"] = ref
print("\n=== (2) pi~ by income segment and PD bin (percent); blank = fewer than 30 applicants or 5 expected wins ===")
print((S.loc[[0.085, 0.105, 0.125, 0.145, 0.165, 0.185, 0.205, 0.225, 0.245, 0.265]] * 100).round(1).to_string())
print("\nadjusted break-even pi~/(1-pi~) by segment (percent):")
A = S / (1 - S)
print((A.loc[[0.105, 0.125, 0.145, 0.165, 0.185, 0.205, 0.225, 0.245]] * 100).round(1).to_string())
C.to_csv(bpm.OUT / "curse_sensitivity_rules.csv")
S.to_csv(bpm.OUT / "curse_by_segment.csv")
