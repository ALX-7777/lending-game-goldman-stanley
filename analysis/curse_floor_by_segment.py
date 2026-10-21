"""Selected default rate pi~ conditioned on income segment AND predicted PD.

Model, on the logit scale: each segment curve is the pooled curve (output/round1/curse_floor.csv, all segments
together) shifted by a segment-specific amount that varies linearly with logit(PD):
    logit pi~_s(p) = logit pi~_pooled(p) + alpha_s + beta_s * (logit p - logit 0.12).
The shift is estimated by weighted least squares on the probability scale on the past loans of the segment (weights =
probability that we win the applicant when every lender prices on its own model, exact types; target = the diagnostic
full-information PD), over the PD range from three points below the segment threshold up to the last one-point bin with at least 20 expected
wins, and it is held constant beyond that range (the trend is not extrapolated).  Segments with no such range at their
threshold (20-40k, 40k+: thresholds at 25% PD, no wins there) keep the pooled curve.  The curve is floored at the PD.
Outputs: output/round1/curse_floor_by_segment.csv (segment, pd, pi, adjusted_breakeven, offset), figures/figC_curse_by_segment.png
Run from the project folder:  python analysis/curse_floor_by_segment.py"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, y, pF, p1, p2, p3, seg = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
m_arr, be, tau_arr = bpm.m_arr, bpm.be, bpm.tau_arr
SEGS = ["0", "1-5k", "5-10k", "10-20k", "20-40k", "40k+"]
MIN_WINS, P_REF = 20, 0.12
logit = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))  # noqa: E731
expit = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

# win probabilities when everyone prices on its own model (break-even + segment margin), exact types
r1, r2, r3 = be(p1) + m_arr, be(p2) + m_arr, be(p3) + m_arr
wp = np.zeros(n)
for t in range(3):
    wp += ((r2 - PREF * (t == 1)) < np.minimum(r1 - PREF * (t == 0), r3 - PREF * (t == 2))) / 3

# the pooled curve (3-bin smoothed adjusted break-even -> pi~)
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")


def pi_pooled(p):
    a = np.interp(p, floor["pd"], floor["adjusted_breakeven"])
    return a / (1 + a)


GRID = np.round(np.arange(0.005, 0.40, 0.005), 4)
EDGES = np.arange(0.0, 0.405, 0.01)
rows, fits = [], {}
for s in SEGS:
    in_s = seg == s
    tau_s = float(tau_arr[in_s][0])
    lo = max(tau_s - 0.03, 0.03)
    # data range: contiguous run of one-point bins with >= MIN_WINS expected wins, starting at lo
    edge = None
    for b_lo in EDGES[:-1]:
        if b_lo + 0.005 < lo:
            continue
        if wp[in_s & (p2 > b_lo) & (p2 <= b_lo + 0.01)].sum() >= MIN_WINS:
            edge = round(b_lo + 0.01, 3)
        else:
            break
    if edge is None or edge <= tau_s:
        fits[s] = dict(alpha=0.0, beta=0.0, lo=lo, edge=np.nan, wins=0.0, own=False)
    else:
        mk = in_s & (p2 >= lo) & (p2 <= edge)
        w, W = wp[mk], np.sqrt(wp[mk])
        base, x, target = logit(pi_pooled(p2[mk])), logit(p2[mk]) - logit(P_REF), pF[mk]
        X = np.column_stack([np.ones(mk.sum()), x])
        beta0, *_ = np.linalg.lstsq(X * W[:, None], (logit(target) - base) * W, rcond=None)  # logit-scale start
        # weighted least squares on the probability scale: the logit-scale fit targets the mean of logit(p_full),
        # which lies below the logit of the mean when p < 0.5 (Jensen), i.e. it under-states the selected default rate
        obj = lambda b: float((w * (target - expit(base + b[0] + b[1] * x)) ** 2).sum())  # noqa: E731
        beta = minimize(obj, beta0, method="L-BFGS-B").x
        fits[s] = dict(alpha=beta[0], beta=beta[1], lo=lo, edge=edge, wins=w.sum(), own=True)
    f = fits[s]
    for p in GRID:
        if f["own"]:
            pc = np.clip(p, f["lo"], f["edge"])
            off = f["alpha"] + f["beta"] * (logit(pc) - logit(P_REF))
        else:
            off = 0.0
        pi = float(max(expit(logit(pi_pooled(p)) + off), p))
        rows.append({"segment": s, "pd": p, "pi": pi, "adjusted_breakeven": pi / (1 - pi) if pi < 0.995 else 99.0, "offset": off})
F = pd.DataFrame(rows)
F.to_csv(bpm.OUT / "curse_floor_by_segment.csv", index=False)
print("=== Segment shift of the pooled curve:  logit pi~_s = logit pi~_pooled + alpha + beta (logit PD - logit 0.12) ===")
for s, f in fits.items():
    a_, b_, lo_, ed_, wi_ = f["alpha"], f["beta"], f["lo"] * 100, (f["edge"] if f["own"] else 0) * 100, f["wins"]
    if f["own"]:
        print(f"{s:7s}: alpha = {a_:+.3f}, beta = {b_:+.3f}, fitted on PD {lo_:.0f}-{ed_:.0f}% ({wi_:,.0f} expected wins), shift held constant above {ed_:.0f}%")
    else:
        print(f"{s:7s}: pooled curve (no one-point bin with {MIN_WINS} expected wins at the threshold)")

# check against raw bins
CENTRES = [0.085, 0.105, 0.125, 0.145, 0.165, 0.185, 0.205, 0.225, 0.245, 0.265]
print("\n=== pi~ (percent): fitted [raw bin, expected wins] -- pooled curve in the last row ===")
for s in ["0", "1-5k", "5-10k", "10-20k"]:
    line = f"{s:7s}"
    Fs = F[F.segment == s]
    for c in CENTRES:
        mk = (seg == s) & (p2 > c - 0.005) & (p2 <= c + 0.005)
        ne = wp[mk].sum()
        fit = np.interp(c, Fs.pd, Fs.pi) * 100
        if ne >= 5:
            raw = (wp[mk] * pF[mk]).sum() / ne * 100
            line += f" | {c:.3f}: {fit:5.1f} [{raw:5.1f}, {ne:4.0f}]"
        else:
            line += f" | {c:.3f}: {fit:5.1f} [  --  ,{ne:4.0f}]"
    print(line)
print("pooled " + "".join(f" | {c:.3f}: {pi_pooled(c) * 100:5.1f}            " for c in CENTRES))

print("\n=== Adjusted break-even (percent) by segment at a few PD levels (pooled curve for comparison) ===")
T = F.pivot(index="pd", columns="segment", values="adjusted_breakeven")[SEGS]
T["pooled"] = np.interp(T.index, floor["pd"], floor["adjusted_breakeven"])
print((T.loc[[0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30]] * 100).round(1).to_string())
print("\nPD at which the adjusted break-even + 0.05 reaches the cap 0.989, by segment:")
for s in SEGS + ["pooled"]:
    t = T[s]
    cross = t.index[t + 0.05 >= 0.989]
    print(f"  {s:7s}: {cross.min() * 100:.1f}%" if len(cross) else f"  {s:7s}: never below 40% PD")

# ---------------------------------------------------------------- figure
INK, INK2, MUTED, GRID_C, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
cols = {"0": "#0d366b", "1-5k": "#2a78d6", "5-10k": "#eb6834", "10-20k": "#6250d6"}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
for s in ["0", "1-5k", "5-10k", "10-20k"]:
    sub = F[F.segment == s]
    f = fits[s]
    lo_, ed_ = f["lo"] * 100, f["edge"] * 100
    ax.plot(sub.pd * 100, sub.pi * 100, color=cols[s], linewidth=2, label=f"income {s} (shift fitted on PD {lo_:.0f}-{ed_:.0f}%)")
    for b_lo in EDGES[:-1]:
        mk = (seg == s) & (p2 > b_lo) & (p2 <= b_lo + 0.01)
        if wp[mk].sum() >= MIN_WINS:
            ax.plot((b_lo + 0.005) * 100, (wp[mk] * pF[mk]).sum() / wp[mk].sum() * 100, "o", color=cols[s], markersize=4, alpha=0.75)
    ax2.plot(sub.pd * 100, np.minimum(sub.adjusted_breakeven, 0.989) * 100, color=cols[s], linewidth=2, label=f"income {s}")
ax.plot(GRID * 100, pi_pooled(GRID) * 100, color=INK2, linestyle="--", linewidth=1.4, label="pooled curve (all segments)")
ax.plot(GRID * 100, GRID * 100, color=MUTED, linestyle=":", linewidth=1.2, label="predicted PD (no selection)")
ax.set_xlabel("our predicted PD, %"); ax.set_ylabel("default rate of the applicants we win, %")
ax.set_title("A. Default rate of the applicants we win, by income segment (dots: raw bins)", loc="left", fontsize=10)
ax.set_xlim(3, 32); ax.set_ylim(0, 60); ax.legend(loc="upper left")
ax2.plot(GRID * 100, np.minimum(T["pooled"].to_numpy(), 0.989) * 100, color=INK2, linestyle="--", linewidth=1.4, label="pooled curve")
ax2.plot(GRID * 100, np.minimum(GRID / (1 - GRID), 0.989) * 100, color=MUTED, linestyle=":", linewidth=1.2, label="plain break-even")
ax2.set_xlabel("our predicted PD, %"); ax2.set_ylabel("adjusted break-even, %")
ax2.set_title("B. Adjusted break-even by income segment (capped at 98.9%)", loc="left", fontsize=10)
ax2.set_xlim(3, 32); ax2.set_ylim(0, 105); ax2.legend(loc="upper left")
for a_ in (ax, ax2):
    for sp in ["top", "right"]:
        a_.spines[sp].set_visible(False)
    a_.grid(True, color=GRID_C, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "figC_curse_by_segment.png", dpi=150)
print("\nfigure written: output/round1/figures/figC_curse_by_segment.png")
