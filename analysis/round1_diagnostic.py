"""Full analysis of OUR round-1 diagnostic file (type-2 lender layout: id, own, competing1, competing3, borrowertype,
accepted, default, profit).

Usage, from the project folder:
    python analysis/round1_diagnostic.py <path to Round1_Diagnostic_xx.csv> [--scores <csv with id,segment,p2,rate>]
By default the scores are output/round1/new_applications_scored.csv (id, segment, p2) joined with the submitted file
output/round1/rates_round1.csv (id, rate).  Everything is realized (default is observed for all 100,000 applicants).

Sections: A reconciliation with our submission and the choice rule; B our results by PD bin and by segment/part;
C the realized winner's curse vs the curve we priced; D the rivals' offers by type and PD (refusals, deciles,
mark-ups over the break-even, nets/caps); E headroom on the loans we won; F ex-post mark-up experiments (uniform and
per PD bin, rivals fixed); G the loans we lost; H model calibration on the new population.
Outputs: tables on stdout, CSVs and a figure in output/round1/diagnostic/."""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
OUT = HERE / "output" / "round1" / "diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
LOAN, PREF, CAP = 10_000, 0.02, 0.989
BINS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 1.0]
TAU = {"0": 0.12, "1-5k": 0.12, "5-10k": 0.10, "10-20k": 0.08, "20-40k": 0.25, "40k+": 0.25}


def section(title):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)


ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--scores", default=None, help="csv with id, segment, p2, rate (default: our scored file + submission)")
args = ap.parse_args()

D = pd.read_csv(args.path)
need = {"id", "own", "competing1", "competing3", "borrowertype", "accepted", "default", "profit"}
missing_cols = need - set(D.columns)
if missing_cols:
    sys.exit(f"this is not a type-2 lender diagnostic (missing columns {missing_cols}); columns are {list(D.columns)}")
if args.scores:
    S = pd.read_csv(args.scores)[["id", "segment", "p2", "rate"]]
else:
    S = pd.read_csv(HERE / "output" / "round1" / "new_applications_scored.csv")[["id", "segment", "p2"]].merge(
        pd.read_csv(HERE / "output" / "round1" / "rates_round1.csv"), on="id")
M = D.merge(S, on="id", how="left", validate="one_to_one")
assert M.p2.notna().all(), "ids of the diagnostic do not match the scores file"
M["segment"] = M["segment"].astype(str)
M["tau"] = M.segment.map(TAU)
M["band"] = M.p2 > M.tau
M["part"] = np.where(~M.band, "safe", np.where(M.own >= CAP - 1e-9, "cap", "band"))
M["bin"] = pd.cut(M.p2, BINS)
M["be"] = M.p2 / (1 - M.p2)
t = M.borrowertype.to_numpy()
e1 = np.where(M.competing1.notna(), M.competing1 - PREF * (t == 1), np.inf)
e3 = np.where(M.competing3.notna(), M.competing3 - PREF * (t == 3), np.inf)
e2 = np.where(M.own.notna(), M.own - PREF * (t == 2), np.inf)
M["rival_eff_min"] = np.minimum(e1, e3)
M["own_eff"] = e2
M["acc_rec"] = (e2 < M.rival_eff_min) & np.isfinite(e2)
M["tie"] = np.isclose(e2, M.rival_eff_min, atol=1e-9) & np.isfinite(e2)
M["d"] = M["default"].astype(int)
acc = M.accepted == 1

# ------------------------------------------------------------------ A. reconciliation
section("A. Reconciliation with our submission and with the choice rule")
dif = (M.own - M.rate).abs()
print(f"own vs submitted rate: max |diff| {dif.max():.6f}; rows differing by more than 1e-4: {int((dif > 1e-4).sum())}; own missing: {int(M.own.isna().sum())}")
print(f"total profit {M.profit.sum():,.0f} EUR | accepted {int(acc.sum()):,} = {acc.mean()*100:.2f}% market share | realized default among accepted {M.d[acc].mean()*100:.2f}% | among all applicants {M.d.mean()*100:.2f}%")
print(f"profit per accepted loan {M.profit[acc].mean():,.0f} EUR | accepted by borrower type: " + ", ".join(f"type {k}: {acc[t == k].mean()*100:.1f}%" for k in (1, 2, 3)))
print(f"acceptance reconstructed from the three rates and the type: mismatches {int((M.acc_rec != acc).sum())} of {len(M)}; exact ties {int(M.tie.sum())} (of which accepted {int((M.tie & acc).sum())})")
print(f"rival refusals: lender 1 made no offer to {M.competing1.isna().mean()*100:.1f}% of applicants, lender 3 to {M.competing3.isna().mean()*100:.1f}%, both to {(M.competing1.isna() & M.competing3.isna()).mean()*100:.1f}%")

# ------------------------------------------------------------------ B. our results by PD bin and by segment/part
section("B. Our results by PD bin (our model's PD) and by segment and part of the rule")


def summarize(g):
    a = g.accepted == 1
    return pd.Series({"offers": len(g), "our mean rate %": g.own.mean() * 100, "share accepted %": a.mean() * 100,
                      "mean p2 %": g.p2.mean() * 100, "default all %": g.d.mean() * 100, "default accepted %": g.d[a].mean() * 100 if a.any() else np.nan,
                      "profit M": g.profit.sum() / 1e6, "profit per accepted loan": g.profit[a].mean() if a.any() else np.nan,
                      "both rivals refused %": (g.competing1.isna() & g.competing3.isna()).mean() * 100})


B = M.groupby("bin", observed=True).apply(summarize)
print(B.round(1).to_string())
B.to_csv(OUT / "results_by_pd_bin.csv")
P = M.groupby(["segment", "part"], observed=True).apply(summarize)
print("\n" + P.round(1).to_string())
P.to_csv(OUT / "results_by_segment_part.csv")
tot = M.groupby("part").agg(offers=("id", "size"), accepted=("accepted", "sum"), profit_M=("profit", lambda s: s.sum() / 1e6))
tot["share %"] = tot.accepted / tot.offers * 100
tot["profit per accepted loan"] = tot.profit_M * 1e6 / tot.accepted
print("\nby part of the rule:\n" + tot.round(1).to_string())

# ------------------------------------------------------------------ C. realized winner's curse
section("C. The realized winner's curse: default of the loans we won vs our PD and vs the curve we priced for")
floor = pd.read_csv(HERE / "output" / "round1" / "curse_floor.csv")
abe = np.interp(M.p2, floor["pd"], floor["adjusted_breakeven"])
M["pi_priced"] = abe / (1 + abe)
rows = []
for b_, g in M.groupby("bin", observed=True):
    a = g.accepted == 1
    if a.sum() < 30:
        continue
    pa = g.d[a].mean()
    rows.append({"bin": str(b_), "accepted": int(a.sum()), "mean p2 %": g.p2[a].mean() * 100, "default of won loans %": pa * 100,
                 "se": np.sqrt(pa * (1 - pa) / a.sum()) * 100, "curve we priced for %": g.pi_priced[a].mean() * 100,
                 "default of loans we did not win %": g.d[~a].mean() * 100 if (~a).any() else np.nan,
                 "break-even of won pool %": pa / (1 - pa) * 100, "our mean rate on won loans %": g.own[a].mean() * 100,
                 "realized margin per won loan": g.profit[a].mean()})
C = pd.DataFrame(rows).set_index("bin")
print(C.round(1).to_string())
C.to_csv(OUT / "realized_curse.csv")

# ------------------------------------------------------------------ D. the rivals
section("D. The rivals' offers by type and by our PD bin (one random lender of each type per applicant)")
for k, col in [(1, "competing1"), (3, "competing3")]:
    r = M[col]
    rows = []
    for b_, g in M.groupby("bin", observed=True):
        rr = g[col].dropna()
        q = rr.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) * 100 if len(rr) else pd.Series([np.nan] * 5, index=[0.1, 0.25, 0.5, 0.75, 0.9])
        mk = (rr - g.be[rr.index]) * 100
        rows.append({"bin": str(b_), "n": len(g), "no offer %": g[col].isna().mean() * 100, "p10": q[0.1], "p25": q[0.25], "median": q[0.5], "p75": q[0.75], "p90": q[0.9],
                     "share >= 90% %": (rr >= 0.90).mean() * 100 if len(rr) else np.nan, "median mark-up over plain BE (pts)": mk.median() if len(mk) else np.nan,
                     "our median rate": g.own.median() * 100, "our rate below rival median %": (g.own < g[col]).mean() * 100})
    R = pd.DataFrame(rows).set_index("bin")
    print(f"\nlender {k} (rates in %):\n" + R.round(1).to_string())
    R.to_csv(OUT / f"rival{k}_by_pd_bin.csv")
    vc = r.round(4).value_counts().head(12)
    print(f"\nlender {k}: most frequent exact rates (nets, caps, flat rules show up here): " + ", ".join(f"{v:.4f} x{c}" for v, c in vc.items()))
    print(f"lender {k}: distinct rate values {r.nunique():,}; offers at exactly 1.0: {int((r == 1.0).sum())}; at 0.99-1.0: {int(((r >= 0.99) & (r < 1.0)).sum())}; above 0.5: {int((r > 0.5).sum())}")
    # rough de-mixing of the four lenders of this type: quartiles of the mark-up within each bin
    rows = []
    for b_, g in M.groupby("bin", observed=True):
        rr = g[col].dropna()
        mk = (rr - g.be[rr.index])
        if len(mk) < 40:
            continue
        qs = mk.quantile([0.125, 0.375, 0.625, 0.875]) * 100
        rows.append({"bin": str(b_), "lowest quarter of offers: mark-up pts": qs.iloc[0], "2nd": qs.iloc[1], "3rd": qs.iloc[2], "highest quarter": qs.iloc[3]})
    print(f"\nlender {k}: mark-up over the plain break-even (points) at the centre of each quarter of the offers, by bin -- four distinct levels suggest four distinct lenders:\n" + pd.DataFrame(rows).set_index("bin").round(1).to_string())

# ------------------------------------------------------------------ E. headroom on the loans we won
section("E. Headroom: how much more we could have charged on the loans we won (rivals fixed)")
W = M[acc].copy()
W["max_own"] = np.minimum(W.rival_eff_min + PREF * (W.borrowertype == 2) - 1e-4, 1.0)
W.loc[~np.isfinite(W.rival_eff_min), "max_own"] = 1.0  # both rivals refused: anything up to 100% wins
W["headroom"] = W.max_own - W.own
W["profit_at_max"] = ((1 - W.d) * W.max_own - W.d) * LOAN
rows = []
for b_, g in W.groupby("bin", observed=True):
    rows.append({"bin": str(b_), "won loans": len(g), "mean headroom pts": g.headroom.mean() * 100, "median headroom pts": g.headroom.median() * 100,
                 "share with headroom > 5 pts %": (g.headroom > 0.05).mean() * 100, "share > 10 pts %": (g.headroom > 0.10).mean() * 100,
                 "share both rivals refused %": (~np.isfinite(g.rival_eff_min)).mean() * 100,
                 "actual profit M": g.profit.sum() / 1e6, "profit at the max winning rate M": g.profit_at_max.sum() / 1e6})
E = pd.DataFrame(rows).set_index("bin")
print(E.round(2).to_string())
E.to_csv(OUT / "headroom_by_pd_bin.csv")
print(f"\nTOTAL: actual {W.profit.sum()/1e6:.1f} M; with every won loan priced at the highest rate that still wins it: {W.profit_at_max.sum()/1e6:.1f} M "
      f"(an upper bound: perfect information about each rival pair). Mean headroom {W.headroom.mean()*100:.1f} pts; by part: " +
      ", ".join(f"{p}: {W.headroom[W.part == p].mean()*100:.1f} pts on {int((W.part == p).sum()):,} loans" for p in ["safe", "band", "cap"] if (W.part == p).any()))

# ------------------------------------------------------------------ F. ex-post mark-up experiments
section("F. Ex-post experiments: uniform mark-ups on our rates with the rivals and types fixed (realized defaults)")


def realized(own_new):
    e2n = np.where(np.isfinite(own_new), own_new - PREF * (t == 2), np.inf)
    accn = (e2n < M.rival_eff_min.to_numpy()) & np.isfinite(e2n)
    prof = np.where(accn, ((1 - M.d.to_numpy()) * np.nan_to_num(own_new, nan=0.0) - M.d.to_numpy()) * LOAN, 0.0)
    return prof.sum() / 1e6, accn.mean() * 100


own = M.own.to_numpy()
safe_mask, band_mask = (~M.band).to_numpy(), M.band.to_numpy()
rows = []
for k in [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    p_safe, s_safe = realized(np.where(safe_mask, np.minimum(own + k, CAP), own))
    p_band, s_band = realized(np.where(band_mask, np.minimum(own + k, CAP), own))
    p_both, s_both = realized(np.minimum(own + k, CAP))
    rows.append({"mark-up pts": k * 100, "safe part only: profit M": p_safe, "share (safe) %": s_safe, "band only: profit M": p_band, "share (band) %": s_band, "both: profit M": p_both, "share (both) %": s_both})
F = pd.DataFrame(rows).set_index("mark-up pts")
print(F.round(1).to_string())
F.to_csv(OUT / "markup_experiments.csv")
rows = []
for b_, idx in M.groupby("bin", observed=True).groups.items():
    idx = np.asarray(M.index.get_indexer(idx))
    best, best_p = None, -np.inf
    for k in np.arange(-0.02, 0.41, 0.01):
        o = own.copy(); o[idx] = np.minimum(np.maximum(own[idx] + k, 0.005), CAP)
        p, _ = realized(o)
        if p > best_p:
            best, best_p = k, p
    base, _ = realized(own)
    rows.append({"bin": str(b_), "best uniform mark-up in this bin (pts)": best * 100, "gain vs actual (M)": best_p - base})
G = pd.DataFrame(rows).set_index("bin")
print("\nbest mark-up per PD bin (rivals fixed, realized defaults):\n" + G.round(2).to_string())
G.to_csv(OUT / "best_markup_by_pd_bin.csv")

# ------------------------------------------------------------------ G. the loans we lost
section("G. The loans we did not win: were they worth winning?")
L = M[~acc & M.own.notna()].copy()
L["lost_to"] = np.where(np.isfinite(L.rival_eff_min) & (L.rival_eff_min < L.own_eff), np.where(e1[L.index] <= e3[L.index], "lender 1", "lender 3"), "tie or no rival offer")
L["cheaper_raw"] = L.own < np.minimum(L.competing1.fillna(np.inf), L.competing3.fillna(np.inf))  # we were the cheapest in raw rate, lost by preference
L["our_profit_if_won"] = ((1 - L.d) * L.own - L.d) * LOAN
L["rival_rate"] = np.where(L.lost_to == "lender 1", L.competing1, L.competing3)
L["rival_profit"] = ((1 - L.d) * L.rival_rate - L.d) * LOAN
print(f"offers not accepted: {len(L):,}; lost to a cheaper lender 1: {int((L.lost_to == 'lender 1').sum()):,}, to lender 3: {int((L.lost_to == 'lender 3').sum()):,}; "
      f"lost although we were the cheapest raw rate (preference): {int(L.cheaper_raw.sum()):,}")
rows = []
for b_, g in L.groupby("bin", observed=True):
    rows.append({"bin": str(b_), "lost": len(g), "realized default %": g.d.mean() * 100, "our rate %": g.own.mean() * 100, "winning rival rate %": g.rival_rate.mean() * 100,
                 "our profit per loan had we won (EUR)": g.our_profit_if_won.mean(), "total we would have made (M)": g.our_profit_if_won.sum() / 1e6,
                 "rival profit per loan (EUR)": g.rival_profit.mean()})
Lt = pd.DataFrame(rows).set_index("bin")
print(Lt.round(1).to_string())
Lt.to_csv(OUT / "lost_loans_by_pd_bin.csv")

# ------------------------------------------------------------------ H. model check on the new population
section("H. Our PD model on the new population (default is observed for everyone)")
from sklearn.metrics import roc_auc_score  # noqa: E402
print(f"AUC of our PD on all 100,000 new applicants: {roc_auc_score(M.d, M.p2):.3f}")
cal = M.groupby(pd.qcut(M.p2, 10, duplicates="drop"), observed=True).agg(n=("d", "size"), mean_p2=("p2", "mean"), realized=("d", "mean"))
cal[["mean_p2", "realized"]] *= 100
print("calibration by decile of PD (%):\n" + cal.round(2).to_string())

# ------------------------------------------------------------------ figure
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
cent = np.array([(b.left + b.right) / 2 for b in B.index]) * 100
ax = axes[0, 0]
for k, c in [(1, "#2a78d6"), (3, "#6250d6")]:
    R = pd.read_csv(OUT / f"rival{k}_by_pd_bin.csv")
    ax.plot(cent[:len(R)], R["median"], color=c, linewidth=2, label=f"lender {k}: median offer")
    ax.fill_between(cent[:len(R)], R["p25"], R["p75"], color=c, alpha=0.12, label=f"lender {k}: interquartile range")
ax.plot(cent, B["our mean rate %"], color="#eb6834", linewidth=2.4, label="our mean rate")
ax.set_xlabel("our PD, %"); ax.set_ylabel("rate, %"); ax.set_ylim(0, 100); ax.set_title("A. Rival offers vs ours, by our PD", loc="left"); ax.legend(loc="upper left")
ax = axes[0, 1]
ax.bar(cent, B["share accepted %"], width=np.diff(np.array(BINS) * 100)[:len(cent)] * 0.8, color="#9ec5f4", label="share of applicants we won, %")
ax2 = ax.twinx(); ax2.plot(cent, B["profit per accepted loan"], color="#eb6834", marker="o", linewidth=2, label="profit per loan won, EUR")
ax.set_xlabel("our PD, %"); ax.set_ylabel("share won, %"); ax2.set_ylabel("EUR per loan won"); ax.set_title("B. Share won and profit per loan, by our PD", loc="left")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
ax = axes[1, 0]
ax.plot(C["mean p2 %"], C["default of won loans %"], color="#e34948", marker="o", linewidth=2, label="realized default of the loans we won")
ax.plot(C["mean p2 %"], C["curve we priced for %"], color="#6250d6", linestyle="--", linewidth=1.6, label="curve we priced for (pooled pi~)")
ax.plot(C["mean p2 %"], C["default of loans we did not win %"], color=MUTED, marker="s", markersize=4, linewidth=1.2, label="realized default of the loans we lost")
ax.plot([0, 40], [0, 40], color=MUTED, linestyle=":", linewidth=1, label="our PD (no selection)")
ax.set_xlabel("our PD, %"); ax.set_ylabel("default, %"); ax.set_title("C. The realized winner's curse", loc="left"); ax.legend(loc="upper left")
ax = axes[1, 1]
ax.bar(cent[:len(E)], E["mean headroom pts"], width=np.diff(np.array(BINS) * 100)[:len(E)] * 0.8, color="#5598e7", label="mean headroom on loans we won, pts")
ax.set_xlabel("our PD, %"); ax.set_ylabel("points of rate"); ax.set_title("D. Headroom: gap to the best rival offer on the loans we won", loc="left"); ax.legend(loc="upper left")
for a_ in axes.flat:
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRID, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(OUT / "figD_round1_diagnostic.png", dpi=150)
print(f"\nfigure written: {OUT / 'figD_round1_diagnostic.png'}; tables in {OUT}")
