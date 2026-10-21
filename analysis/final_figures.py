"""Figures for the final-rule note.
  figA_curse_and_rule.png : left, the winner's curse per PD level (default of the people we win vs our PD, with the
                            95% upper bound); right, the final schedule for several margins X (safe part, one-point
                            phase-in at the threshold, adjusted break-even + X, cap 0.989), X = 0.05 in bold.
  (figB is figures/fig13_final_margin.png, produced by analysis/final_margin_choice.py)
Run from the project folder:  python analysis/final_figures.py"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
n, y, pF, p1, p2, p3 = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3
m_arr, be = bpm.m_arr, bpm.be
CAP, X_FINAL = 0.989, 0.05

# ---------------------------------------------------------------- the structural curse (same computation as curse_adjusted_breakeven.py)
r1, r2, r3 = be(p1) + m_arr, be(p2) + m_arr, be(p3) + m_arr
winprob = np.zeros(n)
for t in range(3):
    winprob += ((r2 - PREF * (t == 1)) < np.minimum(r1 - PREF * (t == 0), r3 - PREF * (t == 2))) / 3
rows = []
for lo in np.arange(0.0, 0.40, 0.01):
    mk = (p2 > lo) & (p2 <= lo + 0.01)
    w = winprob[mk]
    if mk.sum() < 50 or w.sum() < 1:
        continue
    pi_exp = (w * pF[mk]).sum() / w.sum()
    pi_real = (w * y[mk]).sum() / w.sum()
    se = np.sqrt(max(pi_real * (1 - pi_real), 1e-6) / w.sum())
    rows.append({"pd": lo + 0.005, "p_hat": p2[mk].mean(), "pi": pi_exp, "pi95": min(pi_real + 1.645 * se, 0.99), "share won": w.mean()})
F = pd.DataFrame(rows)
F.to_csv(bpm.OUT / "curse_table_final.csv", index=False)
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")

# ---------------------------------------------------------------- figure A
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
x = F["pd"] * 100
ax.plot(x, F["p_hat"] * 100, color=MUTED, linewidth=1.3, linestyle=":", label="our predicted PD  $\\hat p$  (calibrated on everyone)")
ax.plot(x, F["pi"] * 100, color="#2a78d6", linewidth=2.4, label="$\\tilde\\pi(\\hat p)$: default rate of the applicants we WIN when every lender prices on its own model")
ax.plot(x, F["pi95"] * 100, color="#2a78d6", linewidth=1.2, linestyle="--", label="same, realized defaults, 95% upper bound")
ax.fill_between(x, F["p_hat"] * 100, F["pi"] * 100, color="#2a78d6", alpha=0.12, label="the information-asymmetry penalty  $\\tilde\\pi - \\hat p$")
ax.axhline(50, color="#e34948", linewidth=1, linestyle=":")
ax.text(1, 51.5, "above 50% no rate below the cap breaks even", fontsize=8, color="#e34948")
ax.set_xlabel("our predicted PD, %"); ax.set_ylabel("default rate, %"); ax.set_xlim(0, 37); ax.set_ylim(0, 100)
ax.set_title("A. The winner's curse per PD level (past loans, all three signals visible)", loc="left")
ax.legend(loc="upper left")

tau, m = 0.10, 0.035  # a typical low-income segment (5-10k): threshold 10%, margin 3.5%
pp = np.linspace(0.002, 0.40, 1000)
safe = pp / (1 - pp) + m
fs = pd.read_csv(bpm.OUT / "curse_floor_by_segment.csv")
fs = fs[fs["segment"] == "5-10k"]
adj_seg = np.interp(pp, fs["pd"], fs["adjusted_breakeven"])  # the income-segment curve of 5-10k (analysed, not adopted)
adj = np.interp(pp, floor["pd"], floor["adjusted_breakeven"])  # the pooled curve (submitted)
w = 1 / (1 + np.exp(-(pp - tau - 0.01) / 0.01))  # margin transition centred one point above the threshold, scale one point
ax2.plot(pp * 100, np.minimum(pp / (1 - pp), 1) * 100, color=MUTED, linewidth=1.2, linestyle=":", label="plain break-even  $\\hat p/(1-\\hat p)$")
ax2.plot(pp * 100, np.minimum(adj, CAP) * 100, color="#6250d6", linewidth=1.6, linestyle="--", label="adjusted break-even  $\\tilde\\pi/(1-\\tilde\\pi)$  (X = 0)")
for X, c, lw in [(0.02, "#9ec5f4", 1.5), (0.05, "#eb6834", 3.0), (0.075, "#5598e7", 1.5), (0.10, "#1c5cab", 1.5)]:
    band = np.minimum(adj + X, CAP)
    r = safe + w * (band - safe)
    ax2.plot(pp * 100, np.minimum(r, CAP) * 100, color=c, linewidth=lw, label=f"final rule with X = {X:.3f}" + ("  (chosen)" if X == X_FINAL else ""))
r_seg = safe + w * (np.minimum(adj_seg + X_FINAL, CAP) - safe)
ax2.plot(pp * 100, np.minimum(r_seg, CAP) * 100, color="#eb6834", linewidth=1.1, linestyle=":", label="X = 0.050 on the income-segment curve (analysed, not adopted)")
ax2.axhline(CAP * 100, color=AXIS, linewidth=0.9)
ax2.text(39.5, CAP * 100 + 1.2, "cap 0.989", ha="right", fontsize=8, color=INK2)
ax2.axvline(tau * 100, color=AXIS, linewidth=0.9, linestyle=":")
ax2.text(tau * 100 + 0.4, 62, "threshold τ\n(10% in this segment)", fontsize=8, color=INK2)
ax2.set_xlabel("our predicted PD, %"); ax2.set_ylabel("rate we offer, %"); ax2.set_xlim(0, 40); ax2.set_ylim(0, 105)
ax2.set_title("B. The final schedule for several X (segment 5-10k: threshold 10%, margin 3.5%)", loc="left")
ax2.legend(loc="upper left")
for a_ in (ax, ax2):
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRID, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "figA_curse_and_rule.png", dpi=150)
print("written figures/figA_curse_and_rule.png")
print(F.iloc[::2].round(3).to_string(index=False))
