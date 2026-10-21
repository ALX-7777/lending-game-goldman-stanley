"""For each PD category: our expected profit as a function of the rate we charge, under a given behaviour of
the other two lenders.  Shows (1) the selection curve pi(r) = default rate of the applicants we actually win
when we charge r, (2) the profit curve, (3) the best rate per category, and compares with the model's rate.
Run from the project folder:  python analysis/rate_by_pd_band.py
Outputs: output/round1/figures/fig7_rate_by_band.png and output/round1/rate_by_pd_band.csv
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import LOAN, PREF, US, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, y, pF, p1, p2, p3, seg, seeds = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg, bpm.seeds
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be

BANDS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40)]
RATES = np.round(np.arange(0.06, 1.0001, 0.02), 3)
WORLDS = {
    "others: model + 3%, refuse above 25% PD": (bpm.w_reject(0.25)(p1), bpm.w_reject(0.25)(p3)),
    "others: model + margin growing with risk (3% + 1.0 x (PD - 5%))": (bpm.w_pdmargin(1.0)(p1), bpm.w_pdmargin(1.0)(p3)),
    "others: model + 20 points on the band": (bpm.w_margin(0.20)(p1), bpm.w_margin(0.20)(p3)),
    "others: clone of our rule, net 0.898": (bpm.w_net(0.898)(p1), bpm.w_net(0.898)(p3)),
}
model_rate = be(p2) + m_arr  # what the model + our margin would charge

rows = []
curves = {}
for wname, (r1, r3) in WORLDS.items():
    for lo, hi in BANDS:
        band = (p2 > lo) & (p2 <= hi)
        prof = np.zeros(len(RATES)); pi = np.zeros(len(RATES)); share = np.zeros(len(RATES))
        prof_model = pi_model = share_model = 0.0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            types = draw_types(n, seed)
            comp_min = np.minimum(r1 - PREF * (types == 0) + rng.uniform(0, 1e-9, n),
                                  r3 - PREF * (types == 2) + rng.uniform(0, 1e-9, n))[band]
            fan = (types == US)[band]
            pF_b = pF[band]

            def outcome(r_vec):
                won = (r_vec - PREF * fan) < comp_min
                if not won.any():
                    return 0.0, np.nan, 0.0
                return (LOAN * ((1 - pF_b[won]) * r_vec[won] - pF_b[won])).sum() / 1e6, pF_b[won].mean(), won.mean()

            for i, r in enumerate(RATES):
                a, b, c = outcome(np.full(band.sum(), r))
                prof[i] += a / len(seeds); pi[i] += (0 if np.isnan(b) else b) / len(seeds); share[i] += c / len(seeds)
            a, b, c = outcome(model_rate[band])
            prof_model += a / len(seeds); pi_model += (0 if np.isnan(b) else b) / len(seeds); share_model += c / len(seeds)
        k = int(np.argmax(prof))
        curves[(wname, lo, hi)] = (prof, pi, share)
        rows.append({"world": wname, "PD band": f"{lo:.0%}-{hi:.0%}", "applicants": int(band.sum()),
                     "mean PD (model)": p2[band].mean(), "model rate": model_rate[band].mean(),
                     "at model rate: share won": share_model, "at model rate: default of won": pi_model,
                     "at model rate: break-even for those won": pi_model / (1 - pi_model) if pi_model < 1 else np.nan,
                     "at model rate: profit (M)": prof_model,
                     "best rate": RATES[k], "at best: share won": share[k], "at best: default of won": pi[k],
                     "at best: profit (M)": prof[k]})
T = pd.DataFrame(rows)
T.to_csv(bpm.OUT / "rate_by_pd_band.csv", index=False)
for wname in WORLDS:
    print(f"\n=== {wname} ===")
    print(T[T.world == wname].drop(columns="world").round(3).to_string(index=False))

# ---------------------------------------------------------------- figure: profit vs our rate, per band, two worlds
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})
cols = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
show = list(WORLDS)[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
for ax, wname in zip(axes, show):
    for (lo, hi), c in zip(BANDS, cols):
        prof, pi, share = curves[(wname, lo, hi)]
        ax.plot(RATES * 100, prof, color=c, linewidth=2, label=f"PD {lo:.0%}-{hi:.0%}")
        k = int(np.argmax(prof))
        ax.plot(RATES[k] * 100, prof[k], "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        row = T[(T.world == wname) & (T["PD band"] == f"{lo:.0%}-{hi:.0%}")].iloc[0]
        ax.plot(row["model rate"] * 100, row["at model rate: profit (M)"], "x", color=c, markersize=8, markeredgewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title(wname.replace("others: ", ""), loc="left", fontsize=9.5)
    ax.set_xlabel("the rate we charge this category, %")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].set_ylabel("our expected profit on the category, M€")
axes[0].legend(loc="upper left", ncol=2)
fig.suptitle("Expected profit per PD category as a function of our rate  (× = the model's rate + our margin, ● = best rate)",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig7_rate_by_band.png", dpi=150)
print("\nfigure written: output/round1/figures/fig7_rate_by_band.png")
