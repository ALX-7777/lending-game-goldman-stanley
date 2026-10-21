"""Completeness critic, check 3: how much of the prior ranking is driven by duplicate / near-identical worlds
and by the within-regime uniform weighting; leave-one-regime-out and world-collapsed re-weighting from the CSVs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
ROOT = Path("C:/Users/berna/Documents/introduction-to-finance/project"); OUT = ROOT / "output/round1"
P = pd.read_csv(OUT / "band_matrix_payoff.csv", index_col=0)
best = pd.read_csv(ROOT / "analysis/ultra/band_matrix_best_response.csv", index_col=0)
regime = best["regime"].reindex(P.index)
print("regime counts:", regime.value_counts().to_dict(), "missing:", regime.isna().sum())
REGIMES = ["cheap", "margin", "net", "reject", "mixed", "extra"]
PRIORS = {"skeptical": dict(cheap=0.50, margin=0.10, net=0.10, reject=0.15, mixed=0.10, extra=0.05),
          "moderate": dict(cheap=0.35, margin=0.15, net=0.15, reject=0.15, mixed=0.12, extra=0.08),
          "margin-heavy": dict(cheap=0.25, margin=0.35, net=0.10, reject=0.10, mixed=0.12, extra=0.08),
          "net-heavy": dict(cheap=0.25, margin=0.10, net=0.35, reject=0.10, mixed=0.12, extra=0.08)}
def weights(prior, reg):
    present = [r for r in REGIMES if (reg == r).any()]; tot = sum(prior[r] for r in present)
    w = pd.Series(0.0, index=reg.index)
    for r in present:
        mem = reg.index[reg == r]; w[mem] = prior[r] / tot / len(mem)
    return w
A = ["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828", "flat net 0.700", "flat net 0.989", "model + 0.20"]
# (a) duplicate worlds: rows identical (within 0.05M on every action)
key = P.round(1).apply(lambda r: tuple(r.values), axis=1)
dup = key.duplicated(keep=False)
print("\n=== worlds whose payoff row is identical to another world's (every action within 0.05M) ===")
for k, grp in P[dup].groupby(key[dup]):
    print("  ", list(grp.index))
groups = key.groupby(key).ngroup()
reg_col = regime.copy()
print("\n=== prior means: original vs duplicates collapsed (each distinct payoff row counted once inside its regime) ===")
for pname, prior in PRIORS.items():
    w0 = weights(prior, reg_col)
    # collapsed: weight of a regime split equally over distinct rows, then equally among duplicates of a row
    w1 = pd.Series(0.0, index=P.index)
    present = [r for r in REGIMES if (reg_col == r).any()]; tot = sum(prior[r] for r in present)
    for r in present:
        mem = reg_col.index[reg_col == r]; gs = groups[mem]; ng = gs.nunique()
        for gid, idx in gs.groupby(gs).groups.items():
            w1[list(idx)] = prior[r] / tot / ng / len(idx)
    print(f"  {pname:13s} original : " + " | ".join(f"{a} {float((P[a]*w0).sum()):.2f}" for a in A))
    print(f"  {pname:13s} collapsed: " + " | ".join(f"{a} {float((P[a]*w1).sum()):.2f}" for a in A))
print("\n=== leave-one-regime-out: moderate prior mean when a whole regime is dropped (ranking stability) ===")
for r in REGIMES:
    pr = dict(PRIORS["moderate"]); pr[r] = 0.0
    w = weights(pr, reg_col)
    vals = {a: float((P[a] * w).sum()) for a in A}
    print(f"  drop {r:7s}: " + " | ".join(f"{a} {v:.2f}" for a, v in vals.items()) + f"   -> ramp0.70 - flat0.828 = {vals['ramp 0.70->0.828']-vals['flat net 0.828']:+.2f}")
print("\n=== per-world difference ramp 0.70->0.828 minus flat 0.828 (M), sorted ===")
d = (P["ramp 0.70->0.828"] - P["flat net 0.828"]).sort_values()
print(pd.DataFrame({"regime": regime, "diff": d, "flat": P["flat net 0.828"], "ramp": P["ramp 0.70->0.828"]}).loc[d.index].round(2).to_string())
print(f"\nworlds where ramp < flat: {(d < -0.01).sum()}, sum of losses {d[d < 0].sum():.2f}; worlds where ramp > flat: {(d > 0.01).sum()}, sum of gains {d[d > 0].sum():.2f}")
print("break-even probability: if a 'ramp wins' world of average gain G and a 'ramp loses' world of average loss L: P(win-type) > L/(G+L) =",
      f"{-d[d<0].mean():.2f}/({d[d>0].mean():.2f}+{-d[d<0].mean():.2f}) = {(-d[d<0].mean())/(d[d>0].mean()-d[d<0].mean()):.3f}")
