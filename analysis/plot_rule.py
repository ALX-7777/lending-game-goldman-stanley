"""Our pricing rule as a function of the predicted PD, per income segment, with the break-even curves and the
distribution of the new applicants.  Writes output/round1/figures/fig11_our_rule.png.
Run from the project folder:  python analysis/plot_rule.py [ramp_low ramp_high]  (default 0.45 0.65)"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "round1"
L = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
H = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
rule = json.load(open(OUT / "strategy_round1.json"))["rule"]
scored = pd.read_csv(OUT / "new_applications_scored.csv")
floor = pd.read_csv(OUT / "breakeven_ramp_smart.csv")  # selection-adjusted break-evens fitted vs smart rivals

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
SEG_COL = {"0": "#0d366b", "1-5k": "#1c5cab", "5-10k": "#2a78d6", "10-20k": "#5598e7", "20-40k": "#86b6ef", "40k+": "#b7d3f6"}
SEG_LABEL = {"0": "income 0", "1-5k": "income 1–5k", "5-10k": "income 5–10k", "10-20k": "income 10–20k",
             "20-40k": "income 20–40k", "40k+": "income 40k+"}


def our_rate(p, seg):
    tau, m = rule["tau"][seg], rule["margins"][seg]
    safe = p / (1 - p) + m
    frac = np.clip((p - tau) / max(0.25 - tau, 1e-9), 0, 1)
    band = L + (H - L) * frac
    return np.where(p <= tau, safe, band)


p = np.linspace(0.002, 0.40, 800)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [3, 1.1]}, sharex=True)

# reference curves
ax.plot(p * 100, p / (1 - p) * 100, color=MUTED, linewidth=1.3, linestyle=":", label="plain break-even  p/(1−p)")
mids = [0.11, 0.135, 0.165, 0.20, 0.24, 0.28, 0.35]
ax.plot(np.array(mids) * 100, floor["at it: selection-adjusted break-even"].to_numpy() * 100, "s", color="#6250d6", markersize=6,
        label="selection-adjusted break-even (default of the people we'd actually win)")
tau_lo = rule["tau"]["5-10k"]
ax.plot(p * 100, np.where(p <= tau_lo, np.nan, 82.8), color="#e34948", linewidth=1.5, linestyle="--", label="old flat net 0.828 (first submission)")
ax.plot(p * 100, np.where(p <= tau_lo, np.nan, (0.50 + 0.20 * np.clip((p - tau_lo) / (0.25 - tau_lo), 0, 1)) * 100),
        color="#eb6834", linewidth=1.5, linestyle="--", label="alternative ramp 0.50 → 0.70")

# our rule per segment
for seg in ["0", "1-5k", "5-10k", "10-20k", "20-40k", "40k+"]:
    r = our_rate(p, seg) * 100
    tau = rule["tau"][seg]
    lw = 2.6 if seg == "5-10k" else 1.6
    # draw the safe part and the band part separately so the jump at the threshold is visible
    ax.plot(p[p <= tau] * 100, r[p <= tau], color=SEG_COL[seg], linewidth=lw, label=f"our rule — {SEG_LABEL[seg]} (threshold {tau:.0%}, margin {rule['margins'][seg]:.1%})")
    ax.plot(p[p > tau] * 100, r[p > tau], color=SEG_COL[seg], linewidth=lw)
    ax.plot([tau * 100, tau * 100], [our_rate(np.array([tau]), seg)[0] * 100, L * 100], color=SEG_COL[seg], linewidth=0.8, linestyle=":")

ax.axvspan(8, 12, color=GRID, alpha=0.5, lw=0)
ax.text(10, 97, "thresholds\n(8–12% PD\nbelow 20k income)", ha="center", va="top", fontsize=8, color=INK2)
ax.text(2, 14, "safe part:\nbreak-even + 3–4%", fontsize=9, color=INK2)
ax.text(33, 76.5, f"band: lender-of-last-resort ramp\n{L:.2f} at the threshold → {H:.2f} at 25% PD\n(accepted only when nobody else lends cheaper)",
        fontsize=9, color=INK2, ha="center", va="center")
ax.set_ylabel("rate we offer, %")
ax.set_ylim(0, 100)
ax.set_title(f"Our stage-1 pricing rule as a function of the predicted PD  (band ramp {L:.2f} → {H:.2f})", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

# distribution of the new applicants
bins = np.arange(0, 0.405, 0.01)
low = scored[scored.segment.isin(["0", "1-5k", "5-10k", "10-20k"])]
high = scored[scored.segment.isin(["20-40k", "40k+"])]
ax2.hist([high.p2.clip(upper=0.4) * 100, low.p2.clip(upper=0.4) * 100], bins=bins * 100, stacked=True,
         color=["#86b6ef", "#2a78d6"], label=["income above 20k", "income below 20k"], linewidth=0)
ax2.set_xlabel("predicted probability of default, %")
ax2.set_ylabel("new applicants")
ax2.set_title("Where the 100,000 new applicants sit (45.6% fall in the band)", loc="left")
ax2.legend(loc="upper right")
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)
ax2.grid(True, axis="y", color=GRID, linewidth=0.8); ax2.set_axisbelow(True); ax2.tick_params(length=0)
ax2.set_xlim(0, 40)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig11_our_rule.png", dpi=150, bbox_inches="tight")
print("written output/round1/figures/fig11_our_rule.png")
