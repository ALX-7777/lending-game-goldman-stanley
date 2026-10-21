"""Two-fold check of the income-segment curves: is the segment effect real out of sample?
The past loans are split at random into halves A and B.  On half A we rebuild the pooled curve (one-point bins,
3-bin rolling mean of the adjusted break-even, as in curse_adjusted_breakeven.py) and fit the segment shifts exactly
as in curse_floor_by_segment.py.  On half B we compare, bin by bin in the band region, the raw selected default rate
(weights = exact-type win probabilities, target = diagnostic full-information PD) with (i) the pooled curve of half A
and (ii) the segment curve of half A.  Reported: weighted mean error (bias) and mean absolute error per segment, in
points of default, over the bins with at least 20 expected wins; then the roles of A and B are swapped.
Run from the project folder:  python analysis/curse_floor_by_segment_cv.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3, seg = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
m_arr, be, tau_arr = bpm.m_arr, bpm.be, bpm.tau_arr
SEGS = ["0", "1-5k", "5-10k", "10-20k"]
MIN_WINS, P_REF = 20, 0.12
logit = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))  # noqa: E731
expit = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
EDGES = np.arange(0.0, 0.405, 0.01)

r1, r2, r3 = be(p1) + m_arr, be(p2) + m_arr, be(p3) + m_arr
wp = np.zeros(n)
for t in range(3):
    wp += ((r2 - PREF * (t == 1)) < np.minimum(r1 - PREF * (t == 0), r3 - PREF * (t == 2))) / 3


def pooled_curve(mask):
    """Pooled adjusted break-even curve on a subset, as in curse_adjusted_breakeven.py; returns (pd grid, pi)."""
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mk = mask & (p2 > lo) & (p2 <= hi)
        if mk.sum() < 25:  # half the sample: half the applicant minimum
            continue
        w = wp[mk]
        if w.sum() <= 0:
            continue
        pi = (w * pF[mk]).sum() / w.sum()
        rows.append({"pd": (lo + hi) / 2, "abe": pi / (1 - pi)})
    F = pd.DataFrame(rows)
    F["abe_s"] = F["abe"].rolling(3, center=True, min_periods=1).mean()
    return F["pd"].to_numpy(), (F["abe_s"] / (1 + F["abe_s"])).to_numpy()


def fit_shifts(mask, grid_pd, grid_pi):
    out = {}
    for s in SEGS:
        in_s = mask & (seg == s)
        tau_s = float(tau_arr[seg == s][0])
        lo = max(tau_s - 0.03, 0.03)
        edge = None
        for b_lo in EDGES[:-1]:
            if b_lo + 0.005 < lo:
                continue
            if wp[in_s & (p2 > b_lo) & (p2 <= b_lo + 0.01)].sum() >= MIN_WINS / 2:
                edge = round(b_lo + 0.01, 3)
            else:
                break
        if edge is None or edge <= tau_s:
            out[s] = (0.0, 0.0, lo, np.nan)
            continue
        mk = in_s & (p2 >= lo) & (p2 <= edge)
        w = wp[mk]
        base = logit(np.interp(p2[mk], grid_pd, grid_pi))
        x = logit(p2[mk]) - logit(P_REF)
        target = pF[mk]
        X = np.column_stack([np.ones(mk.sum()), x])
        W = np.sqrt(w)
        b0, *_ = np.linalg.lstsq(X * W[:, None], (logit(target) - base) * W, rcond=None)
        obj = lambda b: float((w * (target - expit(base + b[0] + b[1] * x)) ** 2).sum())  # noqa: E731
        b = minimize(obj, b0, method="L-BFGS-B").x
        out[s] = (b[0], b[1], lo, edge)
    return out


def seg_curve(s, p, shifts, grid_pd, grid_pi):
    a, b, lo, edge = shifts[s]
    base = logit(np.interp(p, grid_pd, grid_pi))
    if np.isnan(edge):
        return expit(base)
    pc = np.clip(p, lo, edge)
    return np.maximum(expit(base + a + b * (logit(pc) - logit(P_REF))), p)


rng = np.random.default_rng(2024)
half = rng.random(n) < 0.5
results = []
for name, train, test in [("fit A, test B", half, ~half), ("fit B, test A", ~half, half)]:
    gp, gpi = pooled_curve(train)
    shifts = fit_shifts(train, gp, gpi)
    for s in SEGS:
        tau_s = float(tau_arr[seg == s][0])
        errs_pool, errs_seg, wts = [], [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            if hi <= tau_s - 0.01:
                continue
            mk = test & (seg == s) & (p2 > lo) & (p2 <= hi)
            w = wp[mk]
            if w.sum() < MIN_WINS / 2:
                continue
            raw = (w * pF[mk]).sum() / w.sum()
            c = (lo + hi) / 2
            errs_pool.append(np.interp(c, gp, gpi) - raw)
            errs_seg.append(float(seg_curve(s, c, shifts, gp, gpi)) - raw)
            wts.append(w.sum())
        wts = np.array(wts)
        ep, es = np.array(errs_pool), np.array(errs_seg)
        results.append({"fold": name, "segment": s, "bins": len(wts), "expected wins": wts.sum(),
                        "bias pooled (pts)": (wts * ep).sum() / wts.sum() * 100, "bias segment (pts)": (wts * es).sum() / wts.sum() * 100,
                        "MAE pooled (pts)": (wts * np.abs(ep)).sum() / wts.sum() * 100, "MAE segment (pts)": (wts * np.abs(es)).sum() / wts.sum() * 100,
                        "alpha": shifts[s][0], "beta": shifts[s][1]})
R = pd.DataFrame(results)
print("=== Out-of-sample check of the segment curves (band region, bins with >= 10 expected wins in the test half) ===")
print(R.round(2).to_string(index=False))
print("\nweighted over both folds and the four segments:")
w_all = R["expected wins"]
for c in ["bias pooled (pts)", "bias segment (pts)", "MAE pooled (pts)", "MAE segment (pts)"]:
    print(f"  {c:20s}: {(R[c] * w_all).sum() / w_all.sum():+.2f}")
R.to_csv(bpm.OUT / "curse_floor_by_segment_cv.csv", index=False)
