"""A ramp derived from selection-adjusted break-evens.

For each PD category in the band and each candidate rate r, average over random rival pairs (analysis/rivals.py):
  Q_c(r) = share of the category we win,  pi_c(r) = default rate of the people we win (full-information PD),
  profit_c(r) = N_c * Q_c(r) * [(1 - pi_c(r)) r - pi_c(r)] * 10,000.
The selection-adjusted break-even is pi_c(r)/(1 - pi_c(r)); the data-driven rate of the category is the r that
maximizes expected profit.  The resulting schedule (one rate per category) is validated on FRESH rival draws
against the hand-made ramp 0.65 -> 0.828, the compromise schedule and the flat net.
Run from the project folder:  python analysis/breakeven_ramp.py [profile: mixed|smart] [n_draws]
Outputs: output/round1/breakeven_ramp[_smart].csv, breakeven_ramp_validation[_smart].csv, figures/fig9_breakeven_ramp[_smart].png
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
from lending_game.simulator import LOAN, PREF, US, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
PROFILE = sys.argv[1] if len(sys.argv) > 1 else "mixed"
N_DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SUFFIX = "" if PROFILE == "mixed" else f"_{PROFILE}"
MIX = rivals.MIXTURES[PROFILE]
FIT_MIX = list(MIX)[0]
EDGES = [0.0, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.40]
RATES = np.round(np.arange(0.10, 1.0001, 0.02), 3)
in_band = p2 > tau_arr
cat = np.digitize(p2, EDGES[1:-1])
cat_names = [f"{EDGES[i]:.0%}-{EDGES[i + 1]:.0%}" for i in range(len(EDGES) - 1)]


def draw(rng, W, p):
    return rivals.draw_rival(rng, p, W, PROFILE, tau_arr, n)[0]


# ---------------------------------------------------------------- 1. fit
rng = np.random.default_rng(11)
K, R = len(cat_names), len(RATES)
tot_profit = np.zeros((K, R)); tot_won = np.zeros((K, R)); tot_pf_won = np.zeros((K, R))
members = [(in_band & (cat == k)) for k in range(K)]
N_cat = np.array([m.sum() for m in members])
print(f"profile '{PROFILE}': fitting on {N_DRAWS} rival pairs ({FIT_MIX})...", flush=True)
for d in range(N_DRAWS):
    r1, r3 = draw(rng, MIX[FIT_MIX], p1), draw(rng, MIX[FIT_MIX], p3)
    types = draw_types(n, int(rng.integers(1 << 30)))
    comp_min = np.minimum(r1 - PREF * (types == 0), r3 - PREF * (types == 2))
    fan = types == US
    for k in range(K):
        mk = members[k]
        if not mk.any():
            continue
        cm, fk, pk = comp_min[mk], fan[mk], pF[mk]
        for i, r in enumerate(RATES):
            won = (r - PREF * fk) < cm
            nw = won.sum()
            if nw:
                tot_won[k, i] += nw
                tot_pf_won[k, i] += pk[won].sum()
                tot_profit[k, i] += LOAN * ((1 - pk[won]) * r - pk[won]).sum()
E_profit = tot_profit / N_DRAWS / 1e6
Q = tot_won / N_DRAWS / np.maximum(N_cat[:, None], 1)
PI = np.where(tot_won > 0, tot_pf_won / np.maximum(tot_won, 1), np.nan)
rows, schedule = [], {}
for k in range(K):
    if N_cat[k] == 0:
        continue
    i = int(np.argmax(E_profit[k]))
    i_model = int(np.argmin(np.abs(RATES - (be(p2[members[k]]) + m_arr[members[k]]).mean())))
    schedule[k] = RATES[i]
    rows.append({"PD category": cat_names[k], "applicants in band": int(N_cat[k]), "mean PD": p2[members[k]].mean(),
                 "plain break-even p/(1-p)": be(p2[members[k]].mean()), "model rate": RATES[i_model],
                 "at model rate: share won": Q[k, i_model], "at model rate: default of won": PI[k, i_model],
                 "at model rate: selection-adjusted break-even": PI[k, i_model] / (1 - PI[k, i_model]),
                 "at model rate: E profit (M)": E_profit[k, i_model],
                 "data-driven rate": RATES[i], "at it: share won": Q[k, i], "at it: default of won": PI[k, i],
                 "at it: selection-adjusted break-even": PI[k, i] / (1 - PI[k, i]),
                 "implied margin over it": RATES[i] - PI[k, i] / (1 - PI[k, i]), "at it: E profit (M)": E_profit[k, i]})
T = pd.DataFrame(rows)
T.to_csv(bpm.OUT / f"breakeven_ramp{SUFFIX}.csv", index=False)
print("\n=== Selection-adjusted break-evens and the data-driven rate per PD category (band only) ===")
print(T.round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. validation on fresh draws
safe = be(p2) + m_arr
r65 = 0.65 + (0.828 - 0.65) * ramp_frac(p2)
sched_step = np.full(n, np.nan)
for k, r in schedule.items():
    sched_step[members[k]] = r
mids = np.array([(EDGES[k] + EDGES[k + 1]) / 2 for k in schedule])
vals = np.array([schedule[k] for k in schedule])
sched_smooth = np.interp(p2, mids, vals)
OURS = {"ramp 0.65->0.828 (hand-made)": np.where(in_band, r65, safe),
        "compromise: model + 0.25 if PD <= 0.15, else ramp": np.where(in_band, np.where(p2 <= 0.15, be(p2) + 0.25, r65), safe),
        "data-driven: best rate per category (steps)": np.where(in_band, sched_step, safe),
        "data-driven: interpolated between categories": np.where(in_band, sched_smooth, safe),
        "flat net 0.828": np.where(in_band, 0.828, safe)}
if PROFILE != "mixed" and rivals.DD is not None:  # the schedule fitted on the other profile, for comparison
    edges0, rates0 = rivals.DD
    idx = np.clip(np.digitize(p2, edges0[1:-1]), 0, len(rates0) - 1)
    OURS["data-driven fitted on the 'mixed' profile"] = np.where(in_band, rates0[idx], safe)


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


records = []
for mname, W in MIX.items():
    rng = np.random.default_rng(99)
    print(f"validating on {N_DRAWS} fresh draws: {mname}...", flush=True)
    for d in range(N_DRAWS):
        r1, r3 = draw(rng, W, p1), draw(rng, W, p3)
        types = draw_types(n, int(rng.integers(1 << 30)))
        for a, (q1, q2, q3) in evaluate(r1, OURS, r3, types).items():
            records.append({"mixture": mname, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["mixture", "action"])
V = pd.DataFrame({"mean": g["profit"].mean(), "P5": g["profit"].quantile(0.05), "median": g["profit"].median(),
                  "P95": g["profit"].quantile(0.95), "P(profit < 5)": g["profit"].apply(lambda s: (s < 5).mean()),
                  "P(rank 1)": g["rank"].apply(lambda s: (s == 1).mean()), "P(rank 3)": g["rank"].apply(lambda s: (s == 3).mean())})
V.to_csv(bpm.OUT / f"breakeven_ramp_validation{SUFFIX}.csv")
for mname in MIX:
    print(f"\n=== validation on fresh draws: {mname} ===")
    print(V.loc[mname].loc[list(OURS)].round(2).to_string())

# ---------------------------------------------------------------- 3. figure
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})
cols = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
for k in schedule:
    ax.plot(RATES * 100, E_profit[k], color=cols[k], linewidth=2, label=f"PD {cat_names[k]}")
    i = int(np.argmax(E_profit[k]))
    ax.plot(RATES[i] * 100, E_profit[k, i], "o", color=cols[k], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.axhline(0, color=AXIS, linewidth=1)
ax.set_xlabel("the rate we charge the category, %")
ax.set_ylabel("expected profit on the category, M€ (averaged over random rivals)")
ax.set_title(f"Expected profit vs rate per PD category over {N_DRAWS} random '{PROFILE}' rival pairs (● = best)", loc="left")
ax.legend(loc="upper left", ncol=2)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(True, axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax = axes[1]
grid_p = np.linspace(0.08, 0.40, 200)
ax.plot(grid_p * 100, (0.65 + (0.828 - 0.65) * np.clip((grid_p - 0.10) / 0.15, 0, 1)) * 100, color="#eb6834", linewidth=2, label="hand-made ramp 0.65→0.828")
ax.step([EDGES[k] * 100 for k in schedule] + [EDGES[max(schedule) + 1] * 100], [schedule[k] * 100 for k in schedule] + [schedule[max(schedule)] * 100],
        where="post", color="#008300", linewidth=2, label="data-driven rate per category")
ax.plot(grid_p * 100, np.interp(grid_p, mids, vals) * 100, color="#008300", linewidth=1.2, linestyle="--", label="data-driven, interpolated")
ax.plot(grid_p * 100, (grid_p / (1 - grid_p)) * 100, color=MUTED, linewidth=1.2, linestyle=":", label="plain break-even p/(1−p)")
sab = [T.loc[T["PD category"] == cat_names[k], "at it: selection-adjusted break-even"].iloc[0] for k in schedule]
ax.plot(mids * 100, np.array(sab) * 100, "s", color="#6250d6", markersize=6, label="selection-adjusted break-even at the chosen rate")
ax.set_xlabel("predicted PD of the applicant, %")
ax.set_ylabel("rate charged, %")
ax.set_title("The band schedules compared", loc="left")
ax.legend(loc="lower right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / f"fig9_breakeven_ramp{SUFFIX}.png", dpi=150)
print(f"\nfigure written: output/round1/figures/fig9_breakeven_ramp{SUFFIX}.png")
