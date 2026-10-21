"""Ex-post optimisation of our rate schedule against the rivals' ACTUAL round-1 offers, and the anatomy of the refusals.

With the diagnostic we know, for every applicant, both rival offers (or refusals), the borrower type and the default.
So for ANY rate r we can say whether we would have won the applicant and what we would have earned.  Section 1 finds,
per PD bin, the rate that maximises realized profit, shows how peaked the profit curve is, and splits the optimum's
profit between uncontested applicants (both rivals refused) and contested ones.  Section 2 studies who gets refused:
refusal rates by our PD, income, employment, our signal being 0, borrower type.  Section 3: default of the uncontested
pool vs the contested pool at the same PD (is the pool nobody wants worse?).  Section 4: the lender-1 net at 0.92 and the
lender-3 step levels.  Section 5: ex-post profit of simple candidate schedules for round 2 (rivals fixed).
Usage, from the project folder:  python analysis/round1_expost_optimum.py <Round1_Diagnostic_xx.csv>
Outputs: output/round1/diagnostic/expost_*.csv, figE_expost.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
OUT = HERE / "output" / "round1" / "diagnostic"
LOAN, PREF, CAP = 10_000, 0.02, 0.989
TAU = {"0": 0.12, "1-5k": 0.12, "5-10k": 0.10, "10-20k": 0.08, "20-40k": 0.25, "40k+": 0.25}
BINS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 1.0]
GRID = np.round(np.concatenate([np.arange(0.01, 0.60, 0.005), np.arange(0.60, 0.99, 0.01), [CAP]]), 4)

D = pd.read_csv(sys.argv[1])
S = pd.read_csv(HERE / "output" / "round1" / "new_applications_scored.csv")[["id", "segment", "p2"]].merge(pd.read_csv(HERE / "output" / "round1" / "rates_round1.csv"), on="id")
A = pd.read_csv(HERE / "NewApplications_Lender2_Round1.csv")[["id", "sex", "employment", "married", "income", "signal2"]]
M = D.merge(S, on="id").merge(A, on="id")
M["segment"] = M.segment.astype(str)
M["bin"] = pd.cut(M.p2, BINS)
M["sig0"] = (M.signal2 == 0)
t = M.borrowertype.to_numpy()
e1 = np.where(M.competing1.notna(), M.competing1 - PREF * (t == 1), np.inf)
e3 = np.where(M.competing3.notna(), M.competing3 - PREF * (t == 3), np.inf)
rmin = np.minimum(e1, e3)                      # best rival effective rate (inf = both refused)
thr = rmin + PREF * (t == 2)                   # we win iff our rate < thr
d = M["default"].to_numpy().astype(float)
own = M.own.to_numpy()
M["uncontested"] = ~np.isfinite(rmin)
M["thr"] = thr


def profit_at(r, mask):
    """Realized profit (EUR) and wins if we offer rate r to the applicants in mask (rivals and types fixed)."""
    win = (r < thr[mask])
    return (win * ((1 - d[mask]) * r - d[mask]) * LOAN).sum(), win.sum(), (win & ~np.isfinite(rmin[mask])).sum(), (d[mask] * win).sum()


# ------------------------------------------------------------------ 1. ex-post optimum per PD bin
print("=" * 110 + "\n1. Ex-post optimal rate per PD bin (rivals and types fixed, realized defaults)\n" + "=" * 110)
rows, curves = [], {}
for b_, g in M.groupby("bin", observed=True):
    mask = M.index.isin(g.index)
    prof = np.array([profit_at(r, mask)[0] for r in GRID])
    curves[str(b_)] = prof / 1e6
    i = int(prof.argmax())
    r_star = GRID[i]
    p_star, w_star, u_star, d_star = profit_at(r_star, mask)
    p_act = g.profit.sum()
    # robustness: profit at r* - 10 pts and at the actual mean rate, and the smallest rate within 10% of the optimum
    i_lo = int(np.argmax(prof >= 0.9 * prof[i]))
    r_unc = profit_at(CAP, mask)
    rows.append({"bin": str(b_), "applicants": len(g), "actual mean rate %": g.own.mean() * 100, "actual profit M": p_act / 1e6, "actual share %": g.accepted.mean() * 100,
                 "best single rate %": r_star * 100, "profit at best M": p_star / 1e6, "share at best %": w_star / len(g) * 100,
                 "of the loans won at best: uncontested %": u_star / max(w_star, 1) * 100, "default at best %": d_star / max(w_star, 1) * 100,
                 "lowest rate within 10% of best %": GRID[i_lo] * 100, "profit at 98.9% (uncontested only) M": r_unc[0] / 1e6,
                 "uncontested applicants %": (~np.isfinite(rmin[mask])).mean() * 100})
T1 = pd.DataFrame(rows).set_index("bin")
print(T1.round(1).to_string())
T1.to_csv(OUT / "expost_optimum_by_pd_bin.csv")
print(f"\nTOTAL actual {M.profit.sum()/1e6:.1f} M | sum of per-bin optima {T1['profit at best M'].sum():.1f} M | everyone at 98.9% (uncontested only) {T1['profit at 98.9% (uncontested only) M'].sum():.1f} M")

# ------------------------------------------------------------------ 2. who gets refused
print("\n" + "=" * 110 + "\n2. Refusals: probability that a rival makes no offer, by applicant characteristics\n" + "=" * 110)
M["ref1"], M["ref3"], M["both"] = M.competing1.isna(), M.competing3.isna(), M.competing1.isna() & M.competing3.isna()
for col in ["bin", "segment", "employment", "sig0", "married", "sex", "borrowertype"]:
    g = M.groupby(col, observed=True).agg(n=("id", "size"), lender1_no_offer=("ref1", "mean"), lender3_no_offer=("ref3", "mean"), both_no_offer=("both", "mean"), mean_p2=("p2", "mean"), default=("default", "mean"))
    for c in ["lender1_no_offer", "lender3_no_offer", "both_no_offer", "mean_p2", "default"]:
        g[c] = (g[c] * 100).round(1)
    print(f"\nby {col}:\n" + g.to_string())
# refusals among the SAFEST applicants: what distinguishes the refused?
safe = M[M.p2 < 0.04]
g = safe.groupby(["employment", "sig0"], observed=True).agg(n=("id", "size"), lender1_no_offer=("ref1", "mean"), lender3_no_offer=("ref3", "mean"))
print("\nsafest applicants (PD < 4%): refusals by employment x (our signal2 == 0):\n" + (g.assign(lender1_no_offer=lambda x: x.lender1_no_offer * 100, lender3_no_offer=lambda x: x.lender3_no_offer * 100)).round(1).to_string())
inc = pd.cut(safe.income, [-1, 0, 5000, 10000, 20000, 40000, 100000, 1e9])
g = safe.groupby(inc, observed=True).agg(n=("id", "size"), lender1_no_offer=("ref1", "mean"), lender3_no_offer=("ref3", "mean"))
print("\nsafest applicants (PD < 4%): refusals by income:\n" + (g.assign(lender1_no_offer=lambda x: x.lender1_no_offer * 100, lender3_no_offer=lambda x: x.lender3_no_offer * 100)).round(1).to_string())
M[["id", "ref1", "ref3", "both"]].to_csv(OUT / "expost_refusals.csv", index=False)

# ------------------------------------------------------------------ 3. is the uncontested pool worse?
print("\n" + "=" * 110 + "\n3. Default of the applicants nobody else wanted vs the contested ones, at the same PD\n" + "=" * 110)
rows = []
for b_, g in M.groupby("bin", observed=True):
    u, c = g[g.uncontested], g[~g.uncontested]
    one = g[(g.ref1 ^ g.ref3)]
    rows.append({"bin": str(b_), "uncontested n": len(u), "default uncontested %": u["default"].mean() * 100 if len(u) else np.nan,
                 "one rival refused n": len(one), "default one refused %": one["default"].mean() * 100 if len(one) else np.nan,
                 "both offered n": len(c) - len(one), "default both offered %": g[~g.ref1 & ~g.ref3]["default"].mean() * 100,
                 "mean p2 %": g.p2.mean() * 100, "break-even of uncontested pool %": (lambda q: q / (1 - q) * 100)(u["default"].mean()) if len(u) else np.nan})
T3 = pd.DataFrame(rows).set_index("bin")
print(T3.round(1).to_string())
T3.to_csv(OUT / "expost_uncontested_default.csv")

# ------------------------------------------------------------------ 4. the net and the steps
print("\n" + "=" * 110 + "\n4. Identifiable rival rules\n" + "=" * 110)
net = np.isclose(M.competing1, 0.9199, atol=5e-4)
g = M.groupby("bin", observed=True).apply(lambda x: pd.Series({"lender-1 offers": x.competing1.notna().sum(), "at the 0.92 net": np.isclose(x.competing1, 0.9199, atol=5e-4).sum(), "net share of lender-1 draws %": np.isclose(x.competing1, 0.9199, atol=5e-4).mean() * 100}))
print("lender 1: the 0.92 net by PD bin (one of the four type-1 lenders nets above a threshold; 25% of draws is one lender):\n" + g.round(1).to_string())
vc = M.competing3.round(4).value_counts()
top = vc[vc >= 60]
print(f"\nlender 3: exact rates used 60+ times (a step schedule by one lender): {', '.join(f'{v:.4f} x{c}' for v, c in top.items())}")
step_mask = M.competing3.round(4).isin(top.index)
print("their PD range (our PD) for the most used levels:\n" + M[step_mask].groupby(M.competing3.round(4)).p2.agg(["size", "min", "median", "max"]).loc[top.index[:8]].round(3).to_string())

# ------------------------------------------------------------------ 5. candidate schedules for round 2, ex post
print("\n" + "=" * 110 + "\n5. Ex-post profit of candidate schedules (rivals and types fixed; an upper bound for round 2 if rivals do not move)\n" + "=" * 110)
tau = M.segment.map(TAU).to_numpy()
p2 = M.p2.to_numpy()
be = p2 / (1 - p2)
floor = pd.read_csv(HERE / "output" / "round1" / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
w = 1 / (1 + np.exp(-(p2 - tau - 0.01) / 0.01))
margins = M.segment.map({"0": 0.04, "1-5k": 0.035, "5-10k": 0.035, "10-20k": 0.03, "20-40k": 0.03, "40k+": 0.03}).to_numpy()


def sched(extra_safe=0.0, band_X=0.05, flat=None, everyone=None):
    if everyone is not None:
        return np.full(len(p2), everyone)
    safe = be + margins + extra_safe
    band = np.minimum(adj + band_X, CAP) if flat is None else np.full(len(p2), flat)
    return np.round(np.minimum(safe + w * (band - safe), CAP), 4)


def realized(r):
    win = r < thr
    prof = (win * ((1 - d) * r - d) * LOAN)
    return prof.sum() / 1e6, win.mean() * 100, prof[~np.isfinite(rmin)].sum() / 1e6, (d * win).sum() / max(win.sum(), 1) * 100


cands = {"submitted rule": own, "safe +5, band +0.05": sched(0.05), "safe +10, band +0.15": sched(0.10, 0.15), "safe +20, band +0.30": sched(0.20, 0.30),
         "safe +30, band flat 0.90": sched(0.30, flat=0.90), "plain break-even + 30 pts everywhere": np.minimum(be + 0.30, CAP),
         "plain break-even + 50 pts everywhere": np.minimum(be + 0.50, CAP), "0.60 for everyone": sched(everyone=0.60), "0.90 for everyone": sched(everyone=0.90),
         "0.989 for everyone": sched(everyone=CAP), "max(0.60, plain BE + 20)": np.minimum(np.maximum(be + 0.20, 0.60), CAP), "max(0.90, plain BE + 20)": np.minimum(np.maximum(be + 0.20, 0.90), CAP),
         "safe unchanged, band flat 0.89": sched(0.0, flat=0.89), "safe +2, band flat 0.89": sched(0.02, flat=0.89), "safe +5, band flat 0.89": sched(0.05, flat=0.89),
         "safe unchanged, band flat 0.75": sched(0.0, flat=0.75), "safe unchanged, band flat 0.60": sched(0.0, flat=0.60), "safe unchanged, band adj + 0.20": sched(0.0, 0.20),
         "safe +10, band flat 0.89": sched(0.10, flat=0.89), "safe +20, band flat 0.89": sched(0.20, flat=0.89), "0.89 for everyone": sched(everyone=0.89)}
rows = []
for k, r in cands.items():
    pm, sh, pu, dw = realized(r)
    rows.append({"schedule": k, "realized profit M": pm, "share %": sh, "profit from uncontested M": pu, "default of won loans %": dw, "mean rate offered %": r.mean() * 100})
T5 = pd.DataFrame(rows).set_index("schedule")
print(T5.round(1).to_string())
T5.to_csv(OUT / "expost_candidates.csv")

# ------------------------------------------------------------------ figure
INK, INK2, MUTED, GRIDC, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": AXIS, "legend.frameon": False, "legend.fontsize": 8})
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(curves)))
for (k, v), c in zip(curves.items(), cols):
    ax.plot(GRID * 100, v, color=c, linewidth=1.6, label=f"PD {k}")
ax.set_xlabel("our rate for the whole bin, %"); ax.set_ylabel("realized profit of the bin, M EUR"); ax.set_title("A. Ex-post profit of a single rate per PD bin (rivals fixed)", loc="left")
ax.legend(ncol=2, fontsize=7)
ax = axes[1]
present = [b for b in pd.cut(M.p2, BINS).cat.categories if str(b) in T1.index]
cent = np.array([(b.left + b.right) / 2 for b in present]) * 100
ax.plot(cent, T1["actual mean rate %"], color="#eb6834", marker="o", linewidth=2, label="our actual mean rate")
ax.plot(cent, T1["best single rate %"], color="#2a78d6", marker="s", linewidth=2, label="ex-post best single rate")
ax.plot(cent, T1["lowest rate within 10% of best %"], color="#2a78d6", linestyle="--", linewidth=1.2, label="lowest rate within 10% of the best profit")
ax.bar(cent, T1["uncontested applicants %"], width=np.array([(b.right - b.left) * 100 for b in present]) * 0.8, color="#9ec5f4", alpha=0.6, label="uncontested applicants (both rivals refused), %")
ax.set_xlabel("our PD, %"); ax.set_ylabel("%"); ax.set_title("B. Best rate vs ours, and the share of applicants nobody else wanted", loc="left"); ax.legend(loc="upper left")
for a_ in axes:
    for s in ["top", "right"]:
        a_.spines[s].set_visible(False)
    a_.grid(True, color=GRIDC, linewidth=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)
fig.tight_layout()
fig.savefig(OUT / "figE_expost.png", dpi=150)
print(f"\nfigure written: {OUT / 'figE_expost.png'}")
