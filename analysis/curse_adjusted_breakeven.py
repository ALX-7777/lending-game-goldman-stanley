"""The price of the information asymmetry, measured per PD level from the past loans.

1. Structural curse: if everyone prices on its own model (p/(1-p) + the same margin), who do we win and how often
   do they default?  For each PD level: Q = share won, pi = default rate of the people we win (full-information PD,
   and realized with a 95% upper bound), penalty = pi - p_hat, adjusted break-even = pi/(1-pi).
2. Schedules "adjusted break-even + margin" (a continuous alternative to the ramp), evaluated against rivals who do the
   same thing as us without a net ("likeus"), and against the other profiles.
Run from the project folder:  python analysis/curse_adjusted_breakeven.py [n_draws]
Outputs: output/round1/curse_floor.csv, curse_adjusted_cross.csv, figures/fig12_curse_floor.png
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
from lending_game.simulator import LOAN, PREF, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, y, pF, p1, p2, p3 = bpm.n, bpm.y, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 150

# ---------------------------------------------------------------- 1. structural curse under symmetric pricing
r1, r2, r3 = be(p1) + m_arr, be(p2) + m_arr, be(p3) + m_arr
winprob = np.zeros(n)
for t in range(3):
    e1 = r1 - PREF * (t == 0)
    e2 = r2 - PREF * (t == 1)
    e3 = r3 - PREF * (t == 2)
    winprob += (e2 < np.minimum(e1, e3)) / 3
edges = np.arange(0.0, 0.405, 0.01)
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mk = (p2 > lo) & (p2 <= hi)
    if mk.sum() < 50:
        continue
    w = winprob[mk]
    n_eff = w.sum()
    pi_exp = (w * pF[mk]).sum() / n_eff
    pi_real = (w * y[mk]).sum() / n_eff
    se = np.sqrt(pi_real * (1 - pi_real) / n_eff)
    rows.append({"pd": (lo + hi) / 2, "applicants": int(mk.sum()), "share won": w.mean(), "mean p_hat": p2[mk].mean(),
                 "pi expected": pi_exp, "pi realized": pi_real, "pi realized 95% upper": min(pi_real + 1.645 * se, 0.99),
                 "penalty (pi - p_hat)": pi_exp - p2[mk].mean(), "plain break-even": be(p2[mk].mean()),
                 "adjusted break-even": pi_exp / (1 - pi_exp),
                 "adjusted break-even 95%": min(pi_real + 1.645 * se, 0.99) / (1 - min(pi_real + 1.645 * se, 0.99))})
F = pd.DataFrame(rows)
# smooth the floor a little (3-bin rolling mean) and extend to PD 40%
F["adjusted_breakeven"] = F["adjusted break-even"].rolling(3, center=True, min_periods=1).mean()
F["adjusted_breakeven_95"] = F["adjusted break-even 95%"].rolling(3, center=True, min_periods=1).mean()
F[["pd", "adjusted_breakeven", "adjusted_breakeven_95"]].to_csv(bpm.OUT / "curse_floor.csv", index=False)
print("=== 1. Structural winner's curse when everyone prices on its own model (break-even + 3-4%) ===")
print(F[["pd", "applicants", "share won", "mean p_hat", "pi expected", "pi realized", "pi realized 95% upper",
         "penalty (pi - p_hat)", "plain break-even", "adjusted break-even", "adjusted break-even 95%"]]
      .iloc[::2].round(3).to_string(index=False))

import rivals  # noqa: E402  (imported after the floor is written, so the 'likeus' profile can read it)

floor_p = np.interp(p2, F["pd"], F["adjusted_breakeven"])
floor95_p = np.interp(p2, F["pd"], F["adjusted_breakeven_95"])
safe = be(p2) + m_arr
in_band = p2 > tau_arr
CANDS = {}
for M in [0.05, 0.10, 0.15, 0.20, 0.30]:
    CANDS[f"adjusted break-even + {M:.2f}"] = np.where(in_band, np.minimum(floor_p + M, 1.0), safe)
CANDS["adjusted break-even (95% upper) + 0.10"] = np.where(in_band, np.minimum(floor95_p + 0.10, 1.0), safe)
CANDS["ramp 0.45->0.65 (step)"] = np.where(in_band, 0.45 + 0.20 * ramp_frac(p2), safe)
CANDS["model + 0.20"] = np.where(in_band, np.minimum(be(p2) + 0.20, 1.0), safe)
dd = rivals.DD_SMART
if dd is not None:
    idx = np.clip(np.digitize(p2, dd[0][1:-1]), 0, len(dd[1]) - 1)
    CANDS["data-driven fitted vs smart"] = np.where(in_band, dd[1][idx], safe)
CANDS["flat net 0.828 (first submission)"] = np.where(in_band, 0.828, safe)

PROFILES = [("likeus", list(rivals.MIXTURES["likeus"])[0]), ("smart", list(rivals.MIXTURES["smart"])[0]),
            ("clones", list(rivals.MIXTURES["clones"])[0]), ("netters", list(rivals.MIXTURES["netters"])[0]),
            ("verysmart", list(rivals.MIXTURES["verysmart"])[0]), ("mixed", "baseline (45/15/15/15/10)"),
            ("modelfollowers", list(rivals.MIXTURES["modelfollowers"])[0])]


def evaluate(r1, r2s, r3, types):
    out = {}
    eff1 = r1 - PREF * (types == 0)
    eff3 = r3 - PREF * (types == 2)
    for a, r2 in r2s.items():
        eff2 = np.where(np.isfinite(r2), r2 - PREF * (types == 1), np.inf)
        E = np.column_stack([eff1, eff2, eff3])
        best = E.min(axis=1)
        choice = np.where(np.isfinite(best), E.argmin(axis=1), -1)
        out[a] = [(LOAN * ((1 - pF[choice == k]) * rk[choice == k] - pF[choice == k])).sum() / 1e6 for k, rk in enumerate([r1, r2, r3])]
    return out


records, band_rates = [], []
for profile, mname in PROFILES:
    W = rivals.MIXTURES[profile][mname]
    rng = np.random.default_rng(4040)
    print(f"{profile}: {N_DRAWS} fresh draws...", flush=True)
    for d in range(N_DRAWS):
        r1_, _, d1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)
        r3_, _, _ = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)
        if profile == "likeus" and d < 50:
            band_rates.append(np.nanmedian(np.where(np.isfinite(r1_) & (p1 > tau_arr), r1_, np.nan)))
        types = draw_types(n, int(rng.integers(1 << 30)))
        for a, (q1, q2, q3) in evaluate(r1_, CANDS, r3_, types).items():
            records.append({"profile": profile, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["action", "profile"])
order = [p for p, _ in PROFILES]
M_ = g["profit"].mean().unstack("profile")[order].loc[list(CANDS)]
M_["average"] = M_.mean(axis=1)
R1 = g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile")[order].loc[list(CANDS)]
R1["average"] = R1.mean(axis=1)
R3 = g["rank"].apply(lambda s: (s == 3).mean()).unstack("profile")[order].loc[list(CANDS)]
R3["average"] = R3.mean(axis=1)
print(f"\n('likeus' rivals' median band rate across draws: {np.median(band_rates):.2f}, range {np.min(band_rates):.2f}-{np.max(band_rates):.2f})")
print("\n=== 2. MEAN expected profit (M EUR): schedule x rival profile ===")
print(M_.round(1).to_string())
print("\n=== P(first) ===")
print(R1.round(2).to_string())
print("\n=== P(last) ===")
print(R3.round(2).to_string())
pd.concat({"mean": M_, "P(first)": R1, "P(last)": R3}, axis=1).to_csv(bpm.OUT / "curse_adjusted_cross.csv")

# ---------------------------------------------------------------- figure
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8.5})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))
x = F["pd"] * 100
ax.plot(x, F["mean p_hat"] * 100, color=MUTED, linewidth=1.3, linestyle=":", label="our predicted PD (calibrated on everyone)")
ax.plot(x, F["pi expected"] * 100, color="#2a78d6", linewidth=2.2, label="default rate of the people we WIN (everyone prices on its own model)")
ax.plot(x, F["pi realized 95% upper"] * 100, color="#2a78d6", linewidth=1.2, linestyle="--", label="same, realized, 95% upper bound")
ax.fill_between(x, F["mean p_hat"] * 100, F["pi expected"] * 100, color="#2a78d6", alpha=0.12, label="the information-asymmetry penalty")
ax.set_xlabel("our predicted PD, %"); ax.set_ylabel("default rate, %")
ax.set_title("The winner's curse per PD level (past loans, all three signals visible)", loc="left")
ax.legend(loc="upper left")
cap = lambda s: np.minimum(s, 1.0) * 100  # noqa: E731  (rates are capped at 100% in the game)
ax2.plot(x, cap(F["plain break-even"]), color=MUTED, linewidth=1.3, linestyle=":", label="plain break-even p/(1−p)")
ax2.plot(x, cap(F["adjusted_breakeven"]), color="#6250d6", linewidth=2.2, label="adjusted break-even (if we win, we still break even)")
ax2.plot(x, cap(F["adjusted_breakeven_95"]), color="#6250d6", linewidth=1.2, linestyle="--", label="adjusted break-even, 95% upper bound")
ax2.set_ylim(0, 105)
ax2.axhline(100, color=AXIS, linewidth=0.8)
ax2.text(39.5, 101.5, "cap (100%)", ha="right", fontsize=8, color=INK2)
for M, c in [(0.05, "#9ec5f4"), (0.10, "#5598e7"), (0.20, "#1c5cab")]:
    ax2.plot(x, np.minimum(F["adjusted_breakeven"] + M, 1.0) * 100, color=c, linewidth=1.6, label=f"schedule: adjusted break-even + {M:.2f}")
tau_typ = 0.10
pp = np.linspace(tau_typ, 0.40, 200)
ax2.plot(pp * 100, (0.45 + 0.20 * np.clip((pp - tau_typ) / 0.15, 0, 1)) * 100, color="#eb6834", linewidth=2, label="ramp 0.45 → 0.65 (for reference)")
ax2.set_xlabel("our predicted PD, %"); ax2.set_ylabel("rate, %")
ax2.set_title("Pricing the asymmetry: adjusted break-even + margin, vs the ramp", loc="left")
ax2.legend(loc="upper left")
for a_ in (ax, ax2):
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRID, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig12_curse_floor.png", dpi=150)
print("\nfigure written: output/round1/figures/fig12_curse_floor.png")
