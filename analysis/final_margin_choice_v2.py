"""Final choice of the margin X, second pass: income-segment floor vs pooled floor, more draws.
Candidates: band = min(adjusted break-even + X, 0.989), with the adjusted break-even taken from the pooled curve
(curse_floor.csv, the file submitted on 13 Oct) or from the income-segment curves (curse_floor_by_segment.csv);
logistic transition centred one point above the segment threshold, scale one point; rates rounded to 4 decimals.
Evaluator: exact applicant types (1/3 each), fair tie-splitting, profits computed with the diagnostic full-information
PD.  Rival profiles and prior as in final_margin_choice.py, fresh seed, 400 rival pairs per profile by default.
Reports means with standard errors, P(first), P(last) with binomial standard errors, the 5th percentile, the paired
difference segment-vs-pooled at equal X, and the declared criterion (max prior-weighted mean s.t. P(last) <= 3%).
Run from the project folder:  python analysis/final_margin_choice_v2.py [n_draws]
Outputs: output/round1/final_margin_choice_v2.csv, final_margin_choice_v2_draws.csv, figures/fig13b_final_margin_v2.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3, seg = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 400
CAP = 0.989
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
adj_pooled = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
fs = pd.read_csv(bpm.OUT / "curse_floor_by_segment.csv")
adj_seg = np.full(n, np.nan)
for s in fs["segment"].unique():
    sub = fs[fs["segment"] == s]
    mk = seg == s
    adj_seg[mk] = np.interp(p2[mk], sub["pd"], sub["adjusted_breakeven"])
assert not np.isnan(adj_seg).any()
safe = be(p2) + m_arr
w = 1 / (1 + np.exp(-(p2 - tau_arr - 0.01) / 0.01))


def schedule(adj, X):
    band = np.minimum(adj + X, CAP)
    return np.round(safe + w * (band - safe), 4)


X_SEG, X_POOL = [0.02, 0.03, 0.05, 0.075, 0.10], [0.03, 0.05, 0.075]
CANDS = {f"segment floor + {X:.3f}": schedule(adj_seg, X) for X in X_SEG}
CANDS.update({f"pooled floor + {X:.3f}": schedule(adj_pooled, X) for X in X_POOL})
PROFILES = {"likeus": 0.40, "smart": 0.20, "clones": 0.10, "verysmart": 0.10, "mixed": 0.10, "netters": 0.05, "modelfollowers": 0.05}


def evaluate_exact(r1, r2s, r3):
    """Expected profits of the three lenders with exact type weights and fair tie-splitting."""
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
                rk_ = np.where(np.isfinite(rk), rk, 0.0)
                profits[k] += (wp * LOAN * ((1 - pF) * rk_ - pF)).sum() / 1e6
        out[a] = profits
    return out


PLOT_ONLY = "plot" in sys.argv  # python analysis/final_margin_choice_v2.py plot -> redraw from the saved draws
if PLOT_ONLY:
    D = pd.read_csv(bpm.OUT / "final_margin_choice_v2_draws.csv")
    N_DRAWS = int(D.groupby(["profile", "action"])["draw"].count().iloc[0])
else:
    records = []
    for profile in PROFILES:
        W = list(rivals.MIXTURES[profile].values())[0]
        rng = np.random.default_rng(6060)
        print(f"{profile}: {N_DRAWS} draws (exact types, fair ties)...", flush=True)
        for d in range(N_DRAWS):
            r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
            r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
            for a, (q1, q2, q3) in evaluate_exact(r1, CANDS, r3).items():
                records.append({"profile": profile, "draw": d, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
    D = pd.DataFrame(records)
    D.to_csv(bpm.OUT / "final_margin_choice_v2_draws.csv", index=False)
g = D.groupby(["action", "profile"])
order, acts = list(PROFILES), list(CANDS)
wts = np.array([PROFILES[p] for p in order])
mean = g["profit"].mean().unstack("profile")[order].loc[acts]
se = (g["profit"].std() / np.sqrt(N_DRAWS)).unstack("profile")[order].loc[acts]
first = g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile")[order].loc[acts]
last = g["rank"].apply(lambda s: (s == 3).mean()).unstack("profile")[order].loc[acts]
p5 = g["profit"].quantile(0.05).unstack("profile")[order].loc[acts]
S = pd.DataFrame(index=acts)
S["mean | prior"] = mean.to_numpy() @ wts
S["se(mean)"] = np.sqrt((se.to_numpy() ** 2) @ (wts ** 2))
S["P(first) | prior"] = first.to_numpy() @ wts
S["P(last) | prior"] = last.to_numpy() @ wts
S["se P(last)"] = np.sqrt((last * (1 - last) / N_DRAWS).to_numpy() @ (wts ** 2))
S["P5 | prior"] = p5.to_numpy() @ wts
S["min mean over profiles"] = mean.min(axis=1)
S["mean | likeus"] = mean["likeus"]
S["P(first) | likeus"] = first["likeus"]
S["P(last) | likeus"] = last["likeus"]
print(f"\n=== Prior-weighted results, {N_DRAWS} rival pairs per profile (likeus 40 / smart 20 / clones 10 / very smart 10 / mixed 10 / netters 5 / model-followers 5) ===")
print(S.round(3).to_string())
for lim in [0.03, 0.05]:
    elig = S[S["P(last) | prior"] <= lim]
    choice = elig["mean | prior"].idxmax() if len(elig) else None
    print(f"\nDeclared criterion, P(last) <= {lim:.0%}: eligible = {list(elig.index)}; CHOICE = {choice}")

# paired comparison segment vs pooled floor at equal X, same draws
print("\n=== Paired difference, segment floor minus pooled floor at equal X (M EUR; mean +/- se over draws; prior-weighted and by profile) ===")
P = D.pivot_table(index=["profile", "draw"], columns="action", values="profit")
for X in X_POOL:
    diff = P[f"segment floor + {X:.3f}"] - P[f"pooled floor + {X:.3f}"]
    by = diff.groupby(level="profile").agg(["mean", "std", "count"])
    by["se"] = by["std"] / np.sqrt(by["count"])
    prior_mean = by.loc[order, "mean"].to_numpy() @ wts
    prior_se = np.sqrt((by.loc[order, "se"].to_numpy() ** 2) @ (wts ** 2))
    parts = "  ".join(f"{p}: {by.loc[p, 'mean']:+.2f}+/-{by.loc[p, 'se']:.2f}" for p in order)
    print(f"X = {X:.3f}: prior-weighted {prior_mean:+.2f} +/- {prior_se:.2f}   | {parts}")
print("\n=== Mean profit by profile (M EUR) ===")
print(mean.round(2).to_string())
print("\n=== P(first) by profile ===")
print(first.round(3).to_string())
print("\n=== P(last) by profile ===")
print(last.round(3).to_string())
pd.concat({"summary": S, "mean": mean, "se": se, "P(first)": first, "P(last)": last, "P5": p5}, axis=1).to_csv(bpm.OUT / "final_margin_choice_v2.csv")

# ---------------------------------------------------------------- figure: the dial X, both floors
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
ns = [f"segment floor + {X:.3f}" for X in X_SEG]
npool = [f"pooled floor + {X:.3f}" for X in X_POOL]
xs, xp = np.array(X_SEG) * 100, np.array(X_POOL) * 100
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.errorbar(xp, S.loc[npool, "mean | prior"], yerr=1.96 * S.loc[npool, "se(mean)"], color="#2a78d6", linewidth=2.2, marker="o", capsize=3, label="prior-weighted mean profit, pooled curve (submitted; 95% CI)")
ax.plot(xs, S.loc[ns, "mean | prior"], color="#2a78d6", linewidth=1.2, marker="o", linestyle="--", label="same, income-segment curve (analysed, not adopted)")
ax.plot(xp, S.loc[npool, "mean | likeus"], color="#6250d6", linewidth=1.8, marker="s", label="mean profit vs rivals who price like us, pooled curve")
ax.plot(xs, S.loc[ns, "mean | likeus"], color="#6250d6", linewidth=1.2, marker="s", linestyle="--", label="same, income-segment curve")
ax.set_xlabel("margin X on top of the adjusted break-even, points"); ax.set_ylabel("expected profit, M EUR")
ax.set_title("Profit as a function of the margin X", loc="left"); ax.legend(loc="upper left")
ax = axes[1]
se_first = np.sqrt((first.loc[npool] * (1 - first.loc[npool]) / N_DRAWS).to_numpy() @ (wts ** 2))
ax.errorbar(xp, S.loc[npool, "P(first) | prior"], yerr=1.96 * se_first, color="#2a78d6", linewidth=2.2, marker="o", capsize=3, label="P(first), prior-weighted, pooled curve (submitted)")
ax.errorbar(xp, S.loc[npool, "P(last) | prior"], yerr=1.96 * S.loc[npool, "se P(last)"], color="#e34948", linewidth=2.2, marker="o", capsize=3, label="P(last), prior-weighted, pooled curve")
ax.plot(xs, S.loc[ns, "P(first) | prior"], color="#2a78d6", linewidth=1.2, marker="o", linestyle="--", label="P(first), income-segment curve")
ax.plot(xs, S.loc[ns, "P(last) | prior"], color="#e34948", linewidth=1.2, marker="o", linestyle="--", label="P(last), income-segment curve")
ax.plot(xp, S.loc[npool, "P(first) | likeus"], color="#6250d6", linewidth=1.5, marker="s", linestyle=":", label="P(first) vs rivals who price like us, pooled curve")
ax.axhline(0.03, color=MUTED, linewidth=0.8, linestyle=":")
ax.text(2.1, 0.035, "P(last) limit 3%", fontsize=8, color=INK2)
ax.set_xlabel("margin X, points"); ax.set_ylabel("probability")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title(f"Rank as a function of the margin X ({N_DRAWS} rival pairs per profile)", loc="left"); ax.legend(loc="center left")
for a_ in axes:
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRID, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig13b_final_margin_v2.png", dpi=150)
print("\nfigure written: output/round1/figures/fig13b_final_margin_v2.png")
