"""Final choice of the margin X in  rate = adjusted break-even(PD) + X  (band), with the corrected evaluator:
exact applicant-type weights (1/3 each), fair tie-splitting (1/n among tied lenders; an adverse variant gives us
no ties), rates rounded to 4 decimals as in the CSV, cap 0.989, one-point sigmoid phase-in at the threshold.
Rival profiles from analysis/rivals.py, fresh draws; a prior over profiles reflecting the team's belief that the
rivals are curse-aware and do not net.  Declared criterion: drop any candidate with a negative mean in a core
profile; keep those within 3% of the best prior-weighted mean; among them maximize prior-weighted P(first).
Run from the project folder:  python analysis/final_margin_choice.py [n_draws]
Outputs: output/round1/final_margin_choice.csv, figures/fig13_final_margin.png
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
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
CAP = 0.989
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
safe = be(p2) + m_arr
in_band = p2 > tau_arr
w_phase = 1 / (1 + np.exp(-(6 * (p2 - tau_arr) / 0.01 - 3)))  # ~0.05 at the threshold, ~0.95 one point above


def schedule(X):
    band = np.minimum(adj + X, CAP)
    r = np.where(in_band, safe + w_phase * (band - safe), safe)
    return np.round(r, 4)


CANDS = {f"adjusted break-even + {X:.3f}": schedule(X) for X in [0.0, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]}
CANDS["ramp 0.45->0.65 (step)"] = np.round(np.where(in_band, 0.45 + 0.20 * ramp_frac(p2), safe), 4)
CANDS["model + 0.20"] = np.round(np.where(in_band, np.minimum(be(p2) + 0.20, 1.0), safe), 4)
CANDS["flat net 0.828 (first submission)"] = np.round(np.where(in_band, 0.828, safe), 4)

PROFILES = {"likeus": 0.40, "smart": 0.20, "clones": 0.10, "verysmart": 0.10, "mixed": 0.10, "netters": 0.05, "modelfollowers": 0.05}


def evaluate_exact(r1, r2s, r3, adverse=False):
    """Expected profits of the three lenders with exact type weights and fair (or adverse) tie-splitting."""
    out = {}
    for a, r2 in r2s.items():
        profits = np.zeros(3)
        for t in range(3):
            E = np.column_stack([r1 - PREF * (t == 0), np.where(np.isfinite(r2), r2 - PREF * (t == 1), np.inf), r3 - PREF * (t == 2)])
            m = E.min(axis=1)
            finite = np.isfinite(m)
            tie = (E <= m[:, None] + 1e-12) & finite[:, None]
            n_min = tie.sum(axis=1)
            if adverse:  # we lose every exact tie
                ours_tied = tie[:, 1] & (n_min > 1)
                tie[ours_tied, 1] = False
                n_min = tie.sum(axis=1)
            for k, rk in enumerate([r1, r2, r3]):
                wp = np.where(tie[:, k], 1.0 / np.maximum(n_min, 1), 0.0) / 3
                rk_ = np.where(np.isfinite(rk), rk, 0.0)
                profits[k] += (wp * LOAN * ((1 - pF) * rk_ - pF)).sum() / 1e6
        out[a] = profits
    return out


records = []
for profile in PROFILES:
    W = list(rivals.MIXTURES[profile].values())[0]
    rng = np.random.default_rng(5050)
    print(f"{profile}: {N_DRAWS} draws (exact types, fair ties)...", flush=True)
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        res = evaluate_exact(r1, CANDS, r3)
        res_adv = evaluate_exact(r1, {a: CANDS[a] for a in ["adjusted break-even + 0.100", "flat net 0.828 (first submission)"]}, r3, adverse=True)
        for a, (q1, q2, q3) in res.items():
            records.append({"profile": profile, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2),
                            "adverse": res_adv[a][1] if a in res_adv else np.nan})
D = pd.DataFrame(records)
g = D.groupby(["action", "profile"])
order = list(PROFILES)
T = {"mean": g["profit"].mean().unstack("profile")[order], "P(first)": g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile")[order],
     "P(last)": g["rank"].apply(lambda s: (s == 3).mean()).unstack("profile")[order], "P5": g["profit"].quantile(0.05).unstack("profile")[order]}
wts = np.array([PROFILES[p] for p in order])
S = pd.DataFrame(index=list(CANDS))
for k, t in T.items():
    t = t.loc[list(CANDS)]
    S[f"{k} | prior"] = t.to_numpy() @ wts
S["min mean over profiles"] = T["mean"].loc[list(CANDS)].min(axis=1)
S["mean | likeus"] = T["mean"].loc[list(CANDS), "likeus"]
S["P(first) | likeus"] = T["P(first)"].loc[list(CANDS), "likeus"]
S["P(last) | likeus"] = T["P(last)"].loc[list(CANDS), "likeus"]
adv = D.dropna(subset=["adverse"]).groupby(["action", "profile"])["adverse"].mean().unstack("profile")[order]
S["mean | prior, ADVERSE ties"] = np.nan
for a in adv.index:
    S.loc[a, "mean | prior, ADVERSE ties"] = adv.loc[a].to_numpy() @ wts

print("\n=== Prior-weighted results (likeus 40 / smart 20 / clones 10 / very smart 10 / mixed 10 / netters 5 / model-followers 5) ===")
print(S.round(3).to_string())
best = S["mean | prior"].max()
elig = S[(S["min mean over profiles"] > 0) & (S["mean | prior"] >= 0.97 * best)]
choice = elig["P(first) | prior"].idxmax()
print(f"\nDeclared criterion: best prior-weighted mean = {best:.2f}; eligible (within 3%): {list(elig.index)}")
print(f"CHOICE (max P(first) among eligible): {choice}")
print("\n=== Mean profit by profile (M EUR) ===")
print(T["mean"].loc[list(CANDS)].round(1).to_string())
print("\n=== P(first) by profile ===")
print(T["P(first)"].loc[list(CANDS)].round(2).to_string())
print("\n=== P(last) by profile ===")
print(T["P(last)"].loc[list(CANDS)].round(2).to_string())
pd.concat({"summary": S, **{k: t.loc[list(CANDS)] for k, t in T.items()}}, axis=1).to_csv(bpm.OUT / "final_margin_choice.csv")

# ---------------------------------------------------------------- figure: the dial X
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
Xs = [0.0, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
names = [f"adjusted break-even + {X:.3f}" for X in Xs]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(np.array(Xs) * 100, S.loc[names, "mean | prior"], color="#2a78d6", linewidth=2.2, marker="o", label="prior-weighted mean profit")
ax.plot(np.array(Xs) * 100, S.loc[names, "mean | likeus"], color="#6250d6", linewidth=1.8, marker="s", label="mean profit, 'like us, no net' rivals")
ax.axhline(S.loc["ramp 0.45->0.65 (step)", "mean | prior"], color="#eb6834", linewidth=1.2, linestyle="--", label="ramp 0.45→0.65 (prior-weighted)")
ax.set_xlabel("margin X on top of the adjusted break-even, points"); ax.set_ylabel("expected profit, M€")
ax.set_title("Profit as a function of the margin X", loc="left"); ax.legend(loc="lower right")
ax = axes[1]
ax.plot(np.array(Xs) * 100, S.loc[names, "P(first) | prior"], color="#2a78d6", linewidth=2.2, marker="o", label="P(first), prior-weighted")
ax.plot(np.array(Xs) * 100, S.loc[names, "P(last) | prior"], color="#e34948", linewidth=2.2, marker="o", label="P(last), prior-weighted")
ax.plot(np.array(Xs) * 100, S.loc[names, "P(first) | likeus"], color="#6250d6", linewidth=1.5, marker="s", linestyle="--", label="P(first), 'like us' rivals")
ax.set_xlabel("margin X, points"); ax.set_ylabel("probability")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Rank as a function of the margin X", loc="left"); ax.legend(loc="center right")
for a_ in axes:
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRID, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig13_final_margin.png", dpi=150)
print("\nfigure written: output/round1/figures/fig13_final_margin.png")
