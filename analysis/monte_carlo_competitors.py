"""Monte Carlo over competitor strategies.

Instead of a hand-picked grid of worlds, each rival is DRAWN from a distribution over plausible behaviours:
  - safe-part margin ~ U(1%, 5%); model quality = logit noise sd ~ U(0, 0.3); lecture formula (p + m) with prob 10%
  - treatment of the risky band, with mixture weights W:
      cheap  : model + safe margin, refuse above a threshold in {none, 20%, 25%, 30%, 35%}
      margin : model + M, M ~ U(8%, 40%), lends to everyone
      reject : refuse above t ~ U(10%, 30%)  (or a max-rate cap ~ U(25%, 50%))
      net    : threshold in {ours, 15%, 20%, 25%}; net level: a round number (1.00/0.99/0.95/0.90/0.85/0.80) with prob 1/2,
               else ~ U(0.60, 0.99)
      ramp   : threshold in {ours, 15%, 20%}; L ~ U(0.50, 0.80), H ~ U(max(L + 0.05, 0.80), 0.99)
The two rivals are drawn independently.  For each candidate schedule of ours we record, per draw, our expected
profit and our rank among the three lenders (expected profits), then summarize the distribution.  The mixture
weights are the one subjective input, so we repeat for several weightings.
Run from the project folder:  python analysis/monte_carlo_competitors.py [n_draws]
Outputs: output/round1/montecarlo_summary.csv, montecarlo_draws.csv, figures/fig8_montecarlo.png
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
n, pF, p1, p2, p3, seg = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3, bpm.seg
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

CANDIDATES = ["flat net 0.828", "ramp 0.65->0.828", "ramp 0.60->0.828", "ramp 0.70->0.828", "flat net 0.700",
              "model + 0.20", "hybrid: model + 0.15 to PD 0.15, then net 0.828", "no offer in band"]
OURS = {a: bpm.ACTIONS[a] for a in CANDIDATES}
r65 = 0.65 + (0.828 - 0.65) * ramp_frac(p2)
OURS["compromise: model + 0.20 if PD <= 0.18, else ramp 0.65->0.828"] = np.where(p2 > tau_arr, np.where(p2 <= 0.18, be(p2) + 0.20, r65), be(p2) + m_arr)
CANDIDATES = list(OURS)

MIXTURES = {  # cheap, margin, reject, net, ramp
    "baseline (45/15/15/15/10)": [0.45, 0.15, 0.15, 0.15, 0.10],
    "naive rivals (65/10/15/7/3)": [0.65, 0.10, 0.15, 0.07, 0.03],
    "margin-adders (30/40/10/12/8)": [0.30, 0.40, 0.10, 0.12, 0.08],
    "sophisticated rivals (25/15/10/30/20)": [0.25, 0.15, 0.10, 0.30, 0.20],
}
REGIMES = ["cheap", "margin", "reject", "net", "ramp"]


def draw_rival(rng, W, p):
    """Draw one rival's behaviour and return (its rates on its own PD estimate p, a short description)."""
    q = np.clip(p, 1e-4, 1 - 1e-4)
    sd = rng.uniform(0, 0.3)
    q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + sd * rng.standard_normal(n))))  # model quality
    m_safe = rng.uniform(0.01, 0.05)
    slide = rng.random() < 0.10
    safe = np.minimum((q + m_safe) if slide else (be(q) + m_safe), 1.0)
    regime = REGIMES[rng.choice(5, p=W)]
    if regime == "cheap":
        t = rng.choice([np.inf, 0.20, 0.25, 0.30, 0.35])
        r = np.where(q <= t, safe, np.inf)
        desc = f"cheap m={m_safe:.3f} t={t}"
    elif regime == "margin":
        M = rng.uniform(0.08, 0.40)
        r = np.where(q <= tau_arr, safe, np.minimum(be(q) + M, 1.0))
        desc = f"margin M={M:.2f}"
    elif regime == "reject":
        if rng.random() < 0.7:
            t = rng.uniform(0.10, 0.30)
            r = np.where(q <= t, safe, np.inf)
            desc = f"reject t={t:.2f}"
        else:
            c = rng.uniform(0.25, 0.50)
            r = np.where(safe <= c, safe, np.inf)
            desc = f"cap c={c:.2f}"
    elif regime == "net":
        thr = rng.choice(["ours", 0.15, 0.20, 0.25])
        tt = tau_arr if thr == "ours" else float(thr)
        X = rng.choice([1.0, 0.99, 0.95, 0.90, 0.85, 0.80]) if rng.random() < 0.5 else rng.uniform(0.60, 0.99)
        r = np.where(q <= tt, safe, X)
        desc = f"net X={X:.3f} thr={thr}"
    else:
        thr = rng.choice(["ours", 0.15, 0.20])
        tt = tau_arr if thr == "ours" else float(thr)
        L = rng.uniform(0.50, 0.80)
        H = rng.uniform(max(L + 0.05, 0.80), 0.99)
        frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
        r = np.where(q <= tt, safe, L + (H - L) * frac)
        desc = f"ramp {L:.2f}->{H:.2f} thr={thr}"
    return r, regime, desc


def evaluate(r1, r2s, r3, types):
    """Expected profits of all three lenders for each of our candidate rate vectors r2s (dict)."""
    out = {}
    eff1 = r1 - PREF * (types == 0)
    eff3 = r3 - PREF * (types == 2)
    for a, r2 in r2s.items():
        eff2 = np.where(np.isfinite(r2), r2 - PREF * (types == 1), np.inf)
        E = np.column_stack([eff1, eff2, eff3])
        best = E.min(axis=1)
        choice = np.where(np.isfinite(best), E.argmin(axis=1), -1)
        profits = []
        for k, rk in enumerate([r1, r2, r3]):
            won = choice == k
            profits.append((LOAN * ((1 - pF[won]) * rk[won] - pF[won])).sum() / 1e6)
        out[a] = profits
    return out


records = []
for mname, W in MIXTURES.items():
    rng = np.random.default_rng(2026)
    print(f"{mname}: drawing {N_DRAWS} rival pairs...", flush=True)
    for d in range(N_DRAWS):
        r1, reg1, d1 = draw_rival(rng, W, p1)
        r3, reg3, d3 = draw_rival(rng, W, p3)
        types = draw_types(n, int(rng.integers(1 << 30)))
        res = evaluate(r1, OURS, r3, types)
        for a, (q1, q2, q3) in res.items():
            records.append({"mixture": mname, "draw": d, "rival1": reg1, "rival3": reg3, "desc1": d1, "desc3": d3,
                            "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2), "rival1 profit": q1, "rival3 profit": q3})
D = pd.DataFrame(records)
D.to_csv(bpm.OUT / "montecarlo_draws.csv", index=False)

g = D.groupby(["mixture", "action"])
S = pd.DataFrame({"mean": g["profit"].mean(), "std": g["profit"].std(), "P5": g["profit"].quantile(0.05),
                  "median": g["profit"].median(), "P95": g["profit"].quantile(0.95),
                  "P(profit < 5)": g["profit"].apply(lambda s: (s < 5).mean()),
                  "P(profit > 20)": g["profit"].apply(lambda s: (s > 20).mean()),
                  "P(rank 1)": g["rank"].apply(lambda s: (s == 1).mean()),
                  "P(rank 3)": g["rank"].apply(lambda s: (s == 3).mean())})
S.to_csv(bpm.OUT / "montecarlo_summary.csv")
for mname in MIXTURES:
    print(f"\n=== {mname} ({N_DRAWS} draws) ===")
    print(S.loc[mname].loc[CANDIDATES].round(2).to_string())
best = S["mean"].unstack(0).idxmax()
print("\nbest schedule by mean, per mixture:\n" + best.to_string())
print("\nbest by P(rank 1), per mixture:\n" + S["P(rank 1)"].unstack(0).idxmax().to_string())

# ---------------------------------------------------------------- figure
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})
COL = {"flat net 0.828": "#2a78d6", "ramp 0.65->0.828": "#eb6834", "ramp 0.60->0.828": "#1baf7a", "ramp 0.70->0.828": "#eda100",
       "flat net 0.700": "#e87ba4", "model + 0.20": "#008300", "hybrid: model + 0.15 to PD 0.15, then net 0.828": "#6250d6",
       "no offer in band": "#898781", "compromise: model + 0.20 if PD <= 0.18, else ramp 0.65->0.828": "#e34948"}
SHORT = {"flat net 0.828": "flat 0.828", "ramp 0.65->0.828": "ramp 0.65→0.828", "ramp 0.60->0.828": "ramp 0.60→0.828",
         "ramp 0.70->0.828": "ramp 0.70→0.828", "flat net 0.700": "flat 0.70", "model + 0.20": "model + 0.20",
         "hybrid: model + 0.15 to PD 0.15, then net 0.828": "hybrid +0.15 / net", "no offer in band": "no offer",
         "compromise: model + 0.20 if PD <= 0.18, else ramp 0.65->0.828": "compromise +0.20 / ramp"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax = axes[0]
base = "baseline (45/15/15/15/10)"
for i, a in enumerate(CANDIDATES):
    v = np.sort(D[(D.mixture == base) & (D.action == a)]["profit"].to_numpy())
    ax.plot(v, np.linspace(0, 1, len(v)), color=COL[a], linewidth=2, label=SHORT[a])
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("our expected profit per draw, M€ (symlog scale)")
ax.set_ylabel("cumulative probability")
ax.set_title(f"Distribution of our profit over {N_DRAWS} random rival pairs — {base}", loc="left")
ax.legend(loc="lower right", ncol=2)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
for ax, col, ttl in [(axes[1], "mean", "mean profit, M€"), (axes[2], "P(rank 1)", "probability of finishing first")]:
    x = np.arange(len(MIXTURES))
    w = 0.8 / len(CANDIDATES)
    for i, a in enumerate(CANDIDATES):
        vals = [S.loc[(m, a), col] for m in MIXTURES]
        ax.bar(x + (i - len(CANDIDATES) / 2 + 0.5) * w, vals, width=w - 0.01, color=COL[a], linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[0] for m in MIXTURES], rotation=15, ha="right", fontsize=8)
    ax.set_title(ttl + " by mixture of rival types", loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig8_montecarlo.png", dpi=150)
print("figure written: output/round1/figures/fig8_montecarlo.png")
