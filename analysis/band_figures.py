"""Figures: the evidence behind the band-price decision.  Writes PNGs to output/round1/figures/.

  fig0_matrix.png      the whole payoff matrix (worlds x shortlisted schedules), as a heatmap
  fig1_where_wins.png  where each schedule wins: vs rival net level, vs rival band margin, vs rejecters
  fig2_priors.png      prior-weighted expected profit and maximum regret
  fig3_risk.png        across-world distribution under the moderate prior (P5 - median - P95)
  fig4_rank.png        probability of finishing first, by competitor regime
  fig5_copycat.png     a rival copies our rule with slightly lower margins / a slightly lower ramp
  fig6_margin.png      the margin curve: expected profit vs our safe-part margin, for rival margins 1-5%
Run from the project folder after band_payoff_matrix.py and band_payoff_ci.py:  python analysis/band_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
from lending_game.simulator import US, FastEvaluator, draw_types, simulate  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = bpm.OUT
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
n, y, pF, p1, p2, p3, seg, seeds = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg, bpm.seeds
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac

# ---------------------------------------------------------------- palette and chrome (reference palette, light mode)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"flat net 0.828": "#2a78d6", "ramp 0.65->0.828": "#eb6834", "ramp 0.60->0.828": "#1baf7a",
          "ramp 0.70->0.828": "#eda100", "flat net 0.700": "#e87ba4", "model + 0.20": "#008300"}
LABEL = {"flat net 0.828": "flat net 0.828 (submitted)", "ramp 0.65->0.828": "ramp 0.65→0.828 (proposed)",
         "ramp 0.60->0.828": "ramp 0.60→0.828", "ramp 0.70->0.828": "ramp 0.70→0.828",
         "flat net 0.700": "flat net 0.70", "model + 0.20": "model + 0.20 in band"}
VIOLET, RED = "#6250d6", "#e34948"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
         "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})


def style(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def meur(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))


P = pd.read_csv(OUT / "band_matrix_payoff.csv", index_col=0)
S = pd.read_csv(OUT / "band_matrix_summary.csv", index_col=0)
RISK = pd.read_csv(OUT / "band_matrix_risk.csv")
regime_of = bpm.regime_of
ACT = list(SERIES)

# ---------------------------------------------------------------- fig 0: the matrix as a heatmap
order = ["cheap", "margin", "reject", "net", "mixed"]
rows = [w for r in order for w in P.index if regime_of[w] == r]
M = P.loc[rows, ACT].clip(upper=60)
fig, ax = plt.subplots(figsize=(8.5, 0.19 * len(rows) + 1.8))
im = ax.imshow(M.to_numpy(), aspect="auto", cmap=ListedColormap(BLUES), vmin=0, vmax=60)
ax.set_xticks(range(len(ACT)))
ax.set_xticklabels([LABEL[a] for a in ACT], rotation=25, ha="right", color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([w.replace("both: ", "") for w in rows], fontsize=6.3, color=INK2)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
start = 0
for r in order:
    k = sum(regime_of[w] == r for w in rows)
    if k:
        ax.axhline(start - 0.5, color=SURFACE, linewidth=2)
        ax.text(len(ACT) - 0.4, start - 0.5 + k / 2, r, ha="left", va="center", fontsize=8, color=INK, fontweight="bold",
                transform=ax.transData, rotation=90)
        start += k
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.10)
cb.set_label("expected profit, M€ (colour clipped at 60)", color=INK2)
cb.outline.set_visible(False)
ax.set_title("Payoff matrix: 92 competitor worlds × our band schedules", loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "fig0_matrix.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 1: where each schedule wins
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
nets = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.83, 0.84, 0.85, 0.90, 0.95, 0.99]
ax = axes[0]
for a in ACT:
    ax.plot(nets, [P.loc[f"both: clone net {X:.2f}", a] for X in nets], color=SERIES[a], linewidth=2, marker="o", markersize=4,
            label=LABEL[a])
ax.set_title("A. Both rivals net the band at level X", loc="left")
ax.set_xlabel("rival net level X")
ax.set_ylabel("our expected profit, M€")
ax.axvspan(0.808, 0.848, color=GRID, alpha=0.6, lw=0)
ax.text(0.828, 1.5, "flat 0.828\n± 2%", ha="center", va="bottom", fontsize=7, color=INK2)
style(ax)
ax = axes[1]
Ms = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
for a in ACT:
    ax.plot(Ms, [P.loc[f"both: model + {M:.2f} in band", a] for M in Ms], color=SERIES[a], linewidth=2, marker="o", markersize=4)
ax.set_title("B. Both rivals lend to the band at model + M", loc="left")
ax.set_xlabel("rival band margin M")
style(ax)
ax = axes[2]
rej = {"reject\n>12% PD": "both: reject above PD 0.12", "reject\n>20%": "both: reject above PD 0.20",
       "reject\n>25%": "both: reject above PD 0.25", "cherry-\npicker": "both: cherry-picker (+1% if PD<10%, else refuse)",
       "refuse\nunemployed": "both: refuse unemployed, base otherwise", "rate cap\n25%": "both: max-rate cap 0.25"}
x = np.arange(len(rej))
wdt = 0.13
for i, a in enumerate(ACT):
    ax.bar(x + (i - 2.5) * wdt, [P.loc[w, a] for w in rej.values()], width=wdt - 0.015, color=SERIES[a], linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(list(rej), fontsize=7.5)
ax.set_title("C. Both rivals refuse part of the band", loc="left")
style(ax)
axes[0].legend(loc="upper left", ncol=1)
fig.suptitle("Where each band schedule wins: our expected profit in three families of competitor behaviour", x=0.01,
             ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig1_where_wins.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: priors and regret
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1.2]})
ax = axes[0]
priors = ["prior skeptical", "prior moderate", "prior margin-heavy", "prior net-heavy"]
x = np.arange(len(priors))
for i, a in enumerate(ACT):
    vals = [S.loc[a, pr] for pr in priors]
    bars = ax.bar(x + (i - 2.5) * wdt, vals, width=wdt - 0.015, color=SERIES[a], linewidth=0, label=LABEL[a])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([p.replace("prior ", "") for p in priors])
ax.set_ylabel("expected profit, M€")
ax.set_title("Prior-weighted expected profit (92 competitor worlds)", loc="left")
ax.legend(loc="upper left", ncol=2)
style(ax)
ax = axes[1]
vals = [S.loc[a, "max regret"] for a in ACT]
bars = ax.barh(range(len(ACT)), vals, color=[SERIES[a] for a in ACT], height=0.6, linewidth=0)
for b, v in zip(bars, vals):
    ax.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(ACT)))
ax.set_yticklabels([LABEL[a] for a in ACT], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("maximum regret over worlds, M€ (lower is better)")
ax.set_title("Worst-case regret", loc="left")
style(ax, grid_axis="x")
fig.tight_layout()
fig.savefig(FIG / "fig2_priors.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 3: risk view under the moderate prior
R = RISK[RISK.prior == "moderate"].set_index("action")
acts3 = [a for a in ACT if a in R.index]
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, a in enumerate(acts3):
    r = R.loc[a]
    ax.plot([r["P5"], r["P95"]], [i, i], color=SERIES[a], linewidth=2.5, solid_capstyle="round")
    ax.plot(r["median"], i, "o", color=SERIES[a], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(r["mean"], i, "D", color=SERIES[a], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.text(r["P95"] + 3, i, f"P(round < 5 M€) = {r['P(payoff < 5)']:.0%}   mean {r['mean']:.0f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(acts3)))
ax.set_yticklabels([LABEL[a] for a in acts3])
ax.invert_yaxis()
ax.set_xlabel("expected profit across worlds, M€ — bar: 5th to 95th percentile, ● median, ◆ mean")
ax.set_title("Across-world risk under the moderate prior", loc="left")
ax.set_xlim(0, 200)
style(ax, grid_axis="x")
fig.tight_layout()
fig.savefig(FIG / "fig3_risk.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 4: probability of finishing first, by regime
print("computing ranks...", flush=True)
rank_rows = []
for name, (regime, f1, f3) in bpm.WORLDS.items():
    r1, r3 = bpm.cases[name]
    for a in ACT:
        for seed in seeds:
            res, _ = simulate(np.column_stack([r1, bpm.ACTIONS[a], r3]), draw_types(n, seed), y, pF, seed)
            pr = [res[k]["profit_expected"] for k in range(3)]
            rank_rows.append({"regime": regime, "action": a, "first": pr[US] >= max(pr)})
RK = pd.DataFrame(rank_rows).groupby(["regime", "action"])["first"].mean().unstack()
fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(order))
for i, a in enumerate(ACT):
    vals = [RK.loc[r, a] for r in order]
    bars = ax.bar(x + (i - 2.5) * wdt, vals, width=wdt - 0.015, color=SERIES[a], linewidth=0, label=LABEL[a])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\n({int((regime_of == r).sum())} worlds)" for r in order])
ax.set_ylim(0, 1.42)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("P(we finish first of the three lenders)")
ax.set_title("Finishing first, by competitor regime (expected profits, 3 type draws)", loc="left")
ax.legend(loc="upper center", ncol=3)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "fig4_rank.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 5: a rival copies us with slightly lower margins
print("simulating copycats...", flush=True)
ramp_of = lambda p: 0.65 + (0.828 - 0.65) * ramp_frac(p)
ours = np.where(p2 > tau_arr, ramp_of(p2), be(p2) + m_arr)
base3 = bpm.w_reject(0.25)(p3)


def copycat(p, delta=0.0, eps=0.0):
    return np.where(p > tau_arr, ramp_of(p) - eps, be(p) + m_arr - delta)


def run(r1, r3):
    acc = np.zeros(3)
    for seed in seeds:
        res, _ = simulate(np.column_stack([r1, ours, r3]), draw_types(n, seed), y, pF, seed)
        acc += [res[k]["profit_expected"] / 1e6 for k in range(3)]
    return acc / len(seeds)


deltas = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]
epss = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]
A1 = np.array([run(copycat(p1, d), base3) for d in deltas])          # one copycat (lender 1) + one base rival
A2 = np.array([run(copycat(p1, d), copycat(p3, d)) for d in deltas])  # two copycats
B1 = np.array([run(copycat(p1, 0, e), base3) for e in epss])
B2 = np.array([run(copycat(p1, 0, e), copycat(p3, 0, e)) for e in epss])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
ax.plot(np.array(deltas) * 100, A1[:, US], color=SERIES["ramp 0.65->0.828"], linewidth=2, marker="o", markersize=4, label="us, one copycat + one base rival")
ax.plot(np.array(deltas) * 100, A1[:, 0], color=VIOLET, linewidth=2, marker="o", markersize=4, label="the copycat (one copycat + one base)")
ax.plot(np.array(deltas) * 100, A2[:, US], color=SERIES["ramp 0.65->0.828"], linewidth=2, linestyle="--", marker="s", markersize=4, label="us, two copycats")
ax.plot(np.array(deltas) * 100, A2[:, 0], color=VIOLET, linewidth=2, linestyle="--", marker="s", markersize=4, label="each copycat, two copycats")
ax.axvline(2, color=AXIS, linewidth=1, linestyle=":")
ax.text(2.05, ax.get_ylim()[1] * 0.98, "2% preference", fontsize=7, color=INK2, va="top")
ax.set_xlabel("how much lower the copycat's safe-part margin is, points")
ax.set_ylabel("expected profit, M€")
ax.set_title("A. Same rule as ours, lower margin on the safe part", loc="left")
ax.legend(loc="center right")
style(ax)
ax = axes[1]
ax.plot(np.array(epss) * 100, B1[:, US], color=SERIES["ramp 0.65->0.828"], linewidth=2, marker="o", markersize=4, label="us, one copycat + one base rival")
ax.plot(np.array(epss) * 100, B1[:, 0], color=VIOLET, linewidth=2, marker="o", markersize=4, label="the copycat")
ax.plot(np.array(epss) * 100, B2[:, US], color=SERIES["ramp 0.65->0.828"], linewidth=2, linestyle="--", marker="s", markersize=4, label="us, two copycats")
ax.plot(np.array(epss) * 100, B2[:, 0], color=VIOLET, linewidth=2, linestyle="--", marker="s", markersize=4, label="each copycat, two copycats")
ax.axvline(2, color=AXIS, linewidth=1, linestyle=":")
ax.set_xlabel("how much lower the copycat's band ramp is, points")
ax.set_title("B. Same rule as ours, ramp shifted down in the band", loc="left")
ax.legend(loc="center right")
style(ax)
fig.suptitle("A rival that copies our rule and undercuts it slightly (our schedule: ramp 0.65→0.828)", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig5_copycat.png", dpi=150)
plt.close(fig)
pd.DataFrame({"delta_pts": np.array(deltas) * 100, "us (1 copycat)": A1[:, US], "copycat (1 copycat)": A1[:, 0],
              "us (2 copycats)": A2[:, US], "copycat (2 copycats)": A2[:, 0]}).to_csv(FIG / "fig5_copycat_margin.csv", index=False)
pd.DataFrame({"eps_pts": np.array(epss) * 100, "us (1 copycat)": B1[:, US], "copycat (1 copycat)": B1[:, 0],
              "us (2 copycats)": B2[:, US], "copycat (2 copycats)": B2[:, 0]}).to_csv(FIG / "fig5_copycat_ramp.csv", index=False)

# ---------------------------------------------------------------- fig 6: the margin curve
print("computing margin curves...", flush=True)
margins = np.arange(0, 0.0801, 0.005)
worlds6 = {f"rivals lend to everyone at +{m:.0%}": (bpm.w_everyone(m)(p1), bpm.w_everyone(m)(p3)) for m in [0.01, 0.02, 0.03, 0.04, 0.05]}
worlds6["rivals: +3%, refuse above 25% PD"] = (bpm.w_reject(0.25)(p1), bpm.w_reject(0.25)(p3))
ev6 = FastEvaluator(worlds6, n, seeds, pF)
curves = {w: [] for w in worlds6}
for m in margins:
    r2 = np.where(p2 > tau_arr, ramp_of(p2), be(p2) + m)
    for w, v in ev6.profits(r2).items():
        curves[w].append(np.mean(v) / 1e6)
fig, ax = plt.subplots(figsize=(9, 4.2))
cols = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
for (w, vals), c in zip(list(curves.items())[:5], cols):
    ax.plot(margins * 100, vals, color=c, linewidth=2, label=w)
    k = int(np.argmax(vals))
    ax.plot(margins[k] * 100, vals[k], "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
w, vals = list(curves.items())[5]
ax.plot(margins * 100, vals, color=SERIES["ramp 0.65->0.828"], linewidth=2, linestyle="--", label=w)
k = int(np.argmax(vals))
ax.plot(margins[k] * 100, vals[k], "o", color=SERIES["ramp 0.65->0.828"], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.axvspan(3, 4, color=GRID, alpha=0.7, lw=0)
ax.text(3.5, ax.get_ylim()[0] + 0.2, "our margins\n(3–4% by segment)", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xlabel("our margin above break-even on the safe part, points (band fixed: ramp 0.65→0.828)")
ax.set_ylabel("expected profit, M€")
ax.set_title("The margin curve: hump-shaped, and the hump moves with the rivals' margin (● = best margin per world)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "fig6_margin.png", dpi=150, bbox_inches="tight")
plt.close(fig)
pd.DataFrame({"margin_pts": margins * 100, **{w: v for w, v in curves.items()}}).to_csv(FIG / "fig6_margin_curves.csv", index=False)

print("copycat (safe margin lower by d points): ours / copycat, one copycat + base:")
for d, a in zip(deltas, A1):
    print(f"  d={d * 100:.1f}: us {a[US]:.2f}, copycat {a[0]:.2f}, base {a[2]:.2f}")
print("copycat (ramp lower by e points): ours / copycat, one copycat + base:")
for e, b in zip(epss, B1):
    print(f"  e={e * 100:.1f}: us {b[US]:.2f}, copycat {b[0]:.2f}, base {b[2]:.2f}")
print("margin curve argmax per world:", {w: f"{margins[int(np.argmax(v))] * 100:.1f}%" for w, v in curves.items()})
print("done:", sorted(p.name for p in FIG.glob("*.png")))
