"""Round-2 scenario engine: candidate rate schedules evaluated on the round-1 applicants, with REALIZED defaults, under
transformations of the rivals' actual round-1 offers (from our diagnostic).  Everything we do not know about round 2 is
in the scenarios; everything we know (who defaults, how the rivals priced) is data.

Scenarios (rival offers of the two random rivals of each applicant):
  S0  exactly as in round 1
  S1  the flat quarter of refusals disappears: in every PD bin, refusals equal to 25% of the bin are replaced by draws from the
      offers the same lender type made in that bin (the absent lender, or the signal-0 refusers, start pricing like the others)
  S2  S1, and the 0.92 net lender lowers its net to 0.80
  S3  S1, and every rival offer rises by 5 points (the market learns it was too cheap)
  S4  S1, and in one draw out of four a rival prices every applicant above 10% PD at 0.84 (someone copies a flat high band)
  S5  no refusal at all any more (all replaced by draws) and every offer +5 points (the toughest plausible market)
Candidates: safe-part mark-up x band level, step at the segment threshold for flat bands, logistic transition for the
curve-based bands, cap 0.989; plus flat rates for everyone and a lower-threshold variant.
Usage, from the project folder:  python analysis/round2_scenarios.py [Round1_Diagnostic_xx.csv]
Output: output/round2/scenarios.csv and a printed ranking (prior-weighted mean over scenarios, worst scenario)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
OUT = HERE / "output" / "round2"
OUT.mkdir(parents=True, exist_ok=True)
LOAN, PREF, CAP = 10_000, 0.02, 0.989
TAU = {"0": 0.12, "1-5k": 0.12, "5-10k": 0.10, "10-20k": 0.08, "20-40k": 0.25, "40k+": 0.25}
MARG = {"0": 0.04, "1-5k": 0.035, "5-10k": 0.035, "10-20k": 0.03, "20-40k": 0.03, "40k+": 0.03}
BINS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 1.0]
WEIGHTS = {"S0 as round 1": 0.15, "S1 flat refusals gone": 0.30, "S2 S1 + net at 0.80": 0.15, "S3 S1 + offers +5": 0.15, "S4 S1 + a 0.84 copycat": 0.15, "S5 no refusals, offers +5": 0.10}

path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "Round_1_report_new" / "Round1_Diagnostic_23.csv")
D = pd.read_csv(path)
S = pd.read_csv(HERE / "output" / "round1" / "new_applications_scored.csv")[["id", "segment", "p2"]]
M = D.merge(S, on="id")
M["segment"] = M.segment.astype(str)
M["bin"] = pd.cut(M.p2, BINS)
n = len(M)
t = M.borrowertype.to_numpy()
d = M["default"].to_numpy().astype(float)
p2 = M.p2.to_numpy()
seg = M.segment.to_numpy()
tau = M.segment.map(TAU).to_numpy()
marg = M.segment.map(MARG).to_numpy()
be = p2 / (1 - p2)
floor = pd.read_csv(HERE / "output" / "round1" / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
w_log = 1 / (1 + np.exp(-(p2 - tau - 0.01) / 0.01))
c1, c3 = M.competing1.to_numpy(), M.competing3.to_numpy()
rng = np.random.default_rng(2026)
bins_arr = M.bin.astype(str).to_numpy()


def fill_refusals(c, frac_of_bin=None):
    """Replace refusals with draws from the same bin's offers of the same lender type.
    frac_of_bin=None: replace all; otherwise replace a random subset equal to frac_of_bin of the bin's applicants."""
    out = c.copy()
    for b in np.unique(bins_arr):
        idx = np.where(bins_arr == b)[0]
        pool = c[idx][~np.isnan(c[idx])]
        ref = idx[np.isnan(c[idx])]
        if len(pool) == 0 or len(ref) == 0:
            continue
        k = len(ref) if frac_of_bin is None else min(len(ref), int(round(frac_of_bin * len(idx))))
        chosen = rng.choice(ref, size=k, replace=False)
        out[chosen] = rng.choice(pool, size=k, replace=True)
    return out


def copycat(c, level=0.84, share=0.25):
    out = c.copy()
    hit = (rng.random(n) < share) & (p2 > 0.10)
    out[hit] = np.where(np.isnan(out[hit]), level, np.minimum(out[hit], level))
    return out


c1_s1, c3_s1 = fill_refusals(c1, 0.25), fill_refusals(c3, 0.25)
net = np.isclose(c1_s1, 0.9199, atol=5e-4)
SCEN = {
    "S0 as round 1": (c1, c3),
    "S1 flat refusals gone": (c1_s1, c3_s1),
    "S2 S1 + net at 0.80": (np.where(net, 0.80, c1_s1), c3_s1),
    "S3 S1 + offers +5": (c1_s1 + 0.05, c3_s1 + 0.05),
    "S4 S1 + a 0.84 copycat": (copycat(c1_s1), copycat(c3_s1)),
    "S5 no refusals, offers +5": (fill_refusals(c1) + 0.05, fill_refusals(c3) + 0.05),
}


def sched(extra_safe=0.0, band="adj+0.05", tau_shift=0.0):
    tt = tau + tau_shift
    safe = be + marg + extra_safe
    if isinstance(band, str) and band.startswith("adj+"):
        level = np.minimum(adj + float(band[4:]), CAP)
        w = 1 / (1 + np.exp(-(p2 - tt - 0.01) / 0.01))
        r = safe + w * (level - safe)
    elif band == "everyone":
        r = np.full(n, extra_safe)  # extra_safe carries the flat rate in this case
    else:
        r = np.where(p2 > tt, float(band), safe)
    return np.round(np.minimum(r, CAP), 4)


def realized(r, cc1, cc3):
    e1 = np.where(~np.isnan(cc1), cc1 - PREF * (t == 1), np.inf)
    e3 = np.where(~np.isnan(cc3), cc3 - PREF * (t == 3), np.inf)
    thr = np.minimum(e1, e3) + PREF * (t == 2)
    win = r < thr
    prof = (win * ((1 - d) * r - d) * LOAN).sum() / 1e6
    return prof, win.mean() * 100, (d * win).sum() / max(win.sum(), 1) * 100


CANDS = {"submitted rule (safe m, band adj+0.05)": sched()}
for x in [0.0, 0.02, 0.05, 0.10, 0.20]:
    for band in ["adj+0.20", 0.60, 0.75, 0.85, 0.89, 0.95]:
        CANDS[f"safe +{x*100:.0f}, band {band}"] = sched(x, band)
CANDS["safe +2, band 0.85, thresholds -2 pts"] = sched(0.02, 0.85, -0.02)
CANDS["safe +2, band 0.85, thresholds -4 pts"] = sched(0.02, 0.85, -0.04)
for band in [0.70, 0.78, 0.80, 0.82]:
    for x in [0.0, 0.02]:
        CANDS[f"safe +{x*100:.0f}, band {band}"] = sched(x, band)
for band in [0.78, 0.85]:
    r = sched(0.0, band)
    CANDS[f"safe +0, band {band}, cap 0.989 above 30% PD"] = np.where(p2 > 0.30, CAP, r)
for lvl in [0.60, 0.85, 0.89]:
    CANDS[f"{lvl:.2f} for everyone"] = sched(lvl, "everyone")

rows = []
for name, r in CANDS.items():
    rec = {"candidate": name}
    for s, (cc1, cc3) in SCEN.items():
        pm, sh, dw = realized(r, cc1, cc3)
        rec[s] = pm
        rec[s + " share"] = sh
    rows.append(rec)
R = pd.DataFrame(rows).set_index("candidate")
P = R[list(SCEN)]
R["prior-weighted mean"] = P.to_numpy() @ np.array([WEIGHTS[s] for s in SCEN])
R["worst scenario"] = P.min(axis=1)
R["mean share %"] = R[[s + " share" for s in SCEN]].mean(axis=1)
show = R[list(SCEN) + ["prior-weighted mean", "worst scenario", "mean share %"]].sort_values("prior-weighted mean", ascending=False)
print("=== Realized profit (M EUR) of candidate schedules on the round-1 applicants, by rival scenario ===")
print(f"scenario weights: {WEIGHTS}\n")
print(show.round(1).to_string())
R.to_csv(OUT / "scenarios.csv")
best = show.index[0]
print(f"\nbest prior-weighted: {best}  ({show.loc[best, 'prior-weighted mean']:.1f} M; worst scenario {show.loc[best, 'worst scenario']:.1f} M)")
robust = show[show["worst scenario"] >= 0.9 * show["worst scenario"].max()]
print(f"candidates within 10% of the best worst-case: {list(robust.index[:6])}")
