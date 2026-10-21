"""Cross-evaluation: every candidate band schedule against every rival profile (fresh draws).
Candidates: the hand-made ramp, the compromise, the flat net, the data-driven schedules fitted on each profile,
and simple round-number ramps suggested by the fits.
Run from the project folder after the breakeven_ramp.py fits:  python analysis/cross_profiles.py [n_draws]
Outputs: output/round1/cross_profiles_{mean,rank1,rank3,p_lt5}.csv, figures/fig10_cross_profiles.png
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
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
in_band = p2 > tau_arr
safe = be(p2) + m_arr


def ramp(L, H):
    return np.where(in_band, L + (H - L) * ramp_frac(p2), safe)


def dd(suffix):
    s = rivals._dd_schedule(suffix)
    if s is None:
        return None
    edges, rates = s
    idx = np.clip(np.digitize(p2, edges[1:-1]), 0, len(rates) - 1)
    return np.where(in_band, rates[idx], safe)


CANDS = {"flat net 0.828": np.where(in_band, 0.828, safe),
         "ramp 0.65->0.828 (hand-made)": ramp(0.65, 0.828),
         "compromise: model+0.25 to PD 15%, then ramp": np.where(in_band, np.where(p2 <= 0.15, be(p2) + 0.25, 0.65 + (0.828 - 0.65) * ramp_frac(p2)), safe),
         "ramp 0.60->0.80": ramp(0.60, 0.80), "ramp 0.55->0.75": ramp(0.55, 0.75), "ramp 0.50->0.70": ramp(0.50, 0.70),
         "ramp 0.45->0.65": ramp(0.45, 0.65)}
for suf, lab in [("", "fitted vs mixed"), ("_smart", "fitted vs smart"), ("_clones", "fitted vs clones"),
                 ("_netters", "fitted vs netters"), ("_verysmart", "fitted vs very smart"), ("_modelfollowers", "fitted vs model-followers")]:
    r = dd(suf)
    if r is not None:
        CANDS[f"data-driven {lab}"] = r
PROFILES = [("mixed", "baseline (45/15/15/15/10)"), ("smart", list(rivals.MIXTURES["smart"])[0]), ("clones", list(rivals.MIXTURES["clones"])[0]),
            ("modelfollowers", list(rivals.MIXTURES["modelfollowers"])[0]), ("netters", list(rivals.MIXTURES["netters"])[0]),
            ("verysmart", list(rivals.MIXTURES["verysmart"])[0])]


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
for profile, mname in PROFILES:
    W = rivals.MIXTURES[profile][mname]
    rng = np.random.default_rng(2027)
    print(f"{profile}: {N_DRAWS} fresh draws...", flush=True)
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        types = draw_types(n, int(rng.integers(1 << 30)))
        for a, (q1, q2, q3) in evaluate(r1, CANDS, r3, types).items():
            records.append({"profile": profile, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["action", "profile"])
tabs = {"mean": g["profit"].mean().unstack("profile"), "rank1": g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile"),
        "rank3": g["rank"].apply(lambda s: (s == 3).mean()).unstack("profile"), "p_lt5": g["profit"].apply(lambda s: (s < 5).mean()).unstack("profile")}
order = [p for p, _ in PROFILES]
for k, t in tabs.items():
    t = t[order].loc[list(CANDS)]
    t["average"] = t.mean(axis=1)
    t["worst"] = t.min(axis=1) if k in ("mean", "rank1") else t.max(axis=1)
    tabs[k] = t
    t.to_csv(bpm.OUT / f"cross_profiles_{k}.csv")
print("\n=== MEAN expected profit (M EUR): candidate x rival profile ===")
print(tabs["mean"].round(1).to_string())
print("\n=== P(we finish FIRST) ===")
print(tabs["rank1"].round(2).to_string())
print("\n=== P(we finish LAST) ===")
print(tabs["rank3"].round(2).to_string())
print("\n=== P(profit < 5 M EUR) ===")
print(tabs["p_lt5"].round(2).to_string())

# ---------------------------------------------------------------- figure: heatmaps of mean and P(first)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS})
BLUES = matplotlib.colors.ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, key, ttl, fmt, vmax in [(axes[0], "mean", "mean expected profit, M€", "{:.0f}", 60), (axes[1], "rank1", "probability of finishing first", "{:.0%}", 1.0)]:
    t = tabs[key][order + ["average"]]
    im = ax.imshow(np.clip(t.to_numpy(dtype=float), 0, vmax), aspect="auto", cmap=BLUES, vmin=0, vmax=vmax)
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(list(t.columns), rotation=20, ha="right")
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(list(t.index), fontsize=8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color=SURFACE if v > vmax * 0.55 else INK)
    ax.set_title(ttl + " — our band schedule (rows) × rival profile (columns)", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(bpm.OUT / "figures" / "fig10_cross_profiles.png", dpi=150)
print("\nfigure written: output/round1/figures/fig10_cross_profiles.png")
