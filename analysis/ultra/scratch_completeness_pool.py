"""Completeness critic, check 1: the abandoned pool by p2 bucket, realized vs expected default,
break-even of the ramp's low end, and the new-applications PD distribution vs the past loans."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("C:/Users/berna/Documents/introduction-to-finance/project")
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment
from lending_game.simulator import LOAN
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
OUT = ROOT / "output/round1"
past = pd.read_csv(ROOT / "PastLoans.csv"); y = past["default"].to_numpy(); n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
rule = strategy["rule"]; tau = np.array([rule["tau"][s] for s in seg]); m = np.array([rule["margins"][s] for s in seg])
be = lambda p: p / (1 - p)
band = p2 > tau
print(f"n={n} band share={band.mean():.3f} band mean p2={p2[band].mean():.3f} mean pF={pF[band].mean():.3f} realized default={y[band].mean():.3f}")
print(f"overall: mean p2={p2.mean():.4f} mean pF={pF.mean():.4f} realized={y.mean():.4f}")
df = pd.DataFrame(dict(seg=seg, band=band, p2=p2, pF=pF, y=y, p1=p1, p3=p3, tau=tau, sex=past["sex"].to_numpy()))
g = df[df.band].groupby("seg").agg(n=("p2", "size"), p2=("p2", "mean"), pF=("pF", "mean"), y=("y", "mean"), tau=("tau", "first"))
g["share_of_band"] = g.n / g.n.sum()
print("\n=== band composition by segment (ramp is identical to the flat net where tau = 0.25) ===")
print(g.round(3).to_string())

def ramp(L, H, s=0.25):
    return np.where(band, L + (H - L) * np.clip((p2 - tau) / np.maximum(s - tau, 1e-9), 0, 1), be(p2) + m)
flat = np.where(band, 0.828, be(p2) + m)
rp = ramp(0.70, 0.828)
bins = [0.0, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30, 0.40, 1.0]
for label, mask in [("both competitors reject above own PD 0.25", (p1 > 0.25) & (p3 > 0.25)),
                    ("both reject above own PD 0.20", (p1 > 0.20) & (p3 > 0.20)),
                    ("both reject above OUR tau (clone worlds)", (p1 > tau) & (p3 > tau)),
                    ("both reject above own PD 0.12 (single t)", (p1 > 0.12) & (p3 > 0.12)),
                    ("both reject above own PD 0.15 (single t)", (p1 > 0.15) & (p3 > 0.15))]:
    pool = band & mask
    q = y[pool].mean(); qe = pF[pool].mean()
    print(f"\n=== abandoned pool: {label}: n={pool.sum()}, share of band={pool.sum()/band.sum():.3f}, mean p2={p2[pool].mean():.3f}, mean pF={qe:.3f}, realized default={q:.3f}")
    print(f"    break-even net on this pool: realized {q/(1-q):.3f}, pF-based {qe/(1-qe):.3f}")
    d = df[pool].copy(); d["bucket"] = pd.cut(d.p2, bins)
    d["flat_rate"] = flat[pool]; d["ramp_rate"] = rp[pool]
    d["flat_real"] = np.where(d.y == 1, -LOAN, LOAN * d.flat_rate); d["ramp_real"] = np.where(d.y == 1, -LOAN, LOAN * d.ramp_rate)
    d["flat_exp"] = LOAN * ((1 - d.pF) * d.flat_rate - d.pF); d["ramp_exp"] = LOAN * ((1 - d.pF) * d.ramp_rate - d.pF)
    gg = d.groupby("bucket", observed=True).agg(n=("p2", "size"), p2=("p2", "mean"), pF=("pF", "mean"), y=("y", "mean"), ramp_rate=("ramp_rate", "mean"),
                                                flat_exp=("flat_exp", "sum"), flat_real=("flat_real", "sum"), ramp_exp=("ramp_exp", "sum"), ramp_real=("ramp_real", "sum"))
    for c in ["flat_exp", "flat_real", "ramp_exp", "ramp_real"]:
        gg[c] = gg[c] / 1e6
    gg["be_real"] = gg.y / (1 - gg.y)
    gg["ramp_loses_money_real"] = gg.ramp_real < 0
    print(gg.round(3).to_string())
    print(f"    totals if we won the whole pool (M): flat exp {gg.flat_exp.sum():.2f} real {gg.flat_real.sum():.2f} | ramp exp {gg.ramp_exp.sum():.2f} real {gg.ramp_real.sum():.2f}")

print("\n=== calibration of pF inside the band, by p2 decile (realized - pF, in points) ===")
d = df[df.band].copy(); d["dec"] = pd.qcut(d.p2, 10, duplicates="drop")
c = d.groupby("dec", observed=True).agg(n=("p2", "size"), p2=("p2", "mean"), pF=("pF", "mean"), y=("y", "mean"))
c["real-pF (pts)"] = 100 * (c.y - c.pF); c["real-p2 (pts)"] = 100 * (c.y - c.p2)
print(c.round(3).to_string())
se = np.sqrt(y[band].mean() * (1 - y[band].mean()) / band.sum())
print(f"band: realized {y[band].mean():.4f} vs pF {pF[band].mean():.4f} vs p2 {p2[band].mean():.4f}; binomial se of realized = {se:.4f}")

print("\n=== sex: does a competitor using sex see something we do not? ===")
zs = np.load(OUT / "oos_probas_sex1.npz", allow_pickle=True)
for k in ["p1", "p2", "p3", "pF"]:
    a, b = z[k], zs[k]
    F = past["sex"].to_numpy() == "F"
    print(f"{k}: corr(sex0,sex1)={np.corrcoef(a,b)[0,1]:.4f} mean|diff|={np.abs(a-b).mean():.4f} mean diff F={(b-a)[F].mean():+.4f} M={(b-a)[~F].mean():+.4f} | in band: mean diff F={(b-a)[F&band].mean():+.4f} M={(b-a)[~F&band].mean():+.4f}")
print("realized default by sex: F", y[past.sex == "F"].mean().round(4), "M", y[past.sex == "M"].mean().round(4), "| in band: F", y[(past.sex == "F") & band].mean().round(4), "M", y[(past.sex == "M") & band].mean().round(4))

print("\n=== new applications vs past loans: is the band the same size? ===")
new = pd.read_csv(OUT / "new_applications_scored.csv")
print("columns:", new.columns.tolist())
rates = pd.read_csv(OUT / "rates_round1.csv")
print("rates columns:", rates.columns.tolist(), "n", len(rates), "NaN", rates.isna().sum().to_dict())
r = rates.iloc[:, 1].to_numpy() if rates.shape[1] >= 2 else rates.iloc[:, 0].to_numpy()
print(f"rates_round1: share at 0.828 (net) = {(np.isclose(r, 0.828)).mean():.3f}; past-loans band share = {band.mean():.3f}; share of rates < 0.828: {(r < 0.828 - 1e-9).mean():.3f}")
cand = [c for c in new.columns if any(k in c.lower() for k in ["p2", "pd", "proba", "prob"])]
print("PD-like columns in new_applications_scored:", cand)
for c in cand[:3]:
    v = new[c].to_numpy(dtype=float)
    print(f"  {c}: mean={np.nanmean(v):.4f}, p10/50/90={np.nanpercentile(v,[10,50,90]).round(4)}  | past OOS p2: mean={p2.mean():.4f}, p10/50/90={np.percentile(p2,[10,50,90]).round(4)}")
if "income" in new.columns:
    segn = income_segment(new["income"]); print("segment shares new:", pd.Series(segn).value_counts(normalize=True).round(3).to_dict()); print("segment shares past:", pd.Series(seg).value_counts(normalize=True).round(3).to_dict())
