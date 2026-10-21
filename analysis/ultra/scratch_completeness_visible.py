"""Completeness critic, check 4: (a) competitors refusing on VISIBLE attributes (uncursed pools = the flat net's best case),
(b) two competitors drawing their last-resort rate independently (the symmetric 'both:' grid understates 'at least one undercuts')."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("C:/Users/berna/Documents/introduction-to-finance/project")
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment
from lending_game.simulator import FastEvaluator
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
OUT = ROOT / "output/round1"
past = pd.read_csv(ROOT / "PastLoans.csv"); y = past["default"].to_numpy(); n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"]); emp = past["employment"].to_numpy(); inc = past["income"].to_numpy()
rule = strategy["rule"]; tau = np.array([rule["tau"][s] for s in seg]); m = np.array([rule["margins"][s] for s in seg])
be = lambda p: p / (1 - p)
band = p2 > tau
def ramp_frac(p, s=0.25): return np.clip((p - tau) / np.maximum(s - tau, 1e-9), 0, 1)
def ours(bandrate): return np.where(band, bandrate, be(p2) + m)
ACT = {"flat net 0.828": ours(np.full(n, 0.828)), "flat net 0.989": ours(np.full(n, 0.989)), "flat net 0.900": ours(np.full(n, 0.90)),
       "ramp 0.70->0.828": ours(0.70 + 0.128 * ramp_frac(p2)), "ramp 0.60->0.828": ours(0.60 + 0.228 * ramp_frac(p2)), "model + 0.20": ours(np.minimum(be(p2) + 0.20, 1.0))}
print("band composition by employment (share of band, realized default in band):")
d = pd.DataFrame(dict(emp=emp, band=band, y=y, p2=p2))
print(d[d.band].groupby("emp").agg(n=("y", "size"), default=("y", "mean"), p2=("p2", "mean")).assign(share=lambda t: t.n / t.n.sum()).round(3).to_string())
print("income < 5k share of band:", round(((inc < 5000) & band).sum() / band.sum(), 3))
def base(p): return np.where(p <= 0.25, be(p) + 0.03, np.inf)
def clone(p, X): return np.where(p <= tau, be(p) + m, X)
cases = {}
for label, mask in [("unemployed", emp == "unemployed"), ("student", emp == "student"), ("unemployed or student", np.isin(emp, ["unemployed", "student"])),
                    ("income < 5k", inc < 5000), ("income = 0", inc == 0)]:
    cases[f"both: no loan if {label}; else +3%, reject >25%"] = (np.where(mask, np.inf, base(p1)), np.where(mask, np.inf, base(p3)))
    cases[f"both: no loan if {label}; else clone net 0.90"] = (np.where(mask, np.inf, clone(p1, 0.90)), np.where(mask, np.inf, clone(p3, 0.90)))
    cases[f"both: no loan if {label}; else clone net 0.80"] = (np.where(mask, np.inf, clone(p1, 0.80)), np.where(mask, np.inf, clone(p3, 0.80)))
# (b) independent pairs on a net grid
G = [0.70, 0.75, 0.80, 0.83, 0.85, 0.90, 0.99]
for a in G:
    for b in G:
        if a <= b:
            cases[f"PAIR {a:.2f}/{b:.2f}"] = (clone(p1, a), clone(p3, b))
ev = FastEvaluator(cases, n, [0, 1, 2], pF)
P = pd.DataFrame({a: {w: float(np.mean(v)) / 1e6 for w, v in ev.profits(r2).items()} for a, r2 in ACT.items()}).loc[list(cases)]
vis = P.loc[[w for w in P.index if w.startswith("both:")]].copy(); vis["ramp - flat"] = vis["ramp 0.70->0.828"] - vis["flat net 0.828"]
print("\n=== (a) refusals on visible attributes (expected profit, M) ===")
print(vis.round(2).to_string())
pair = P.loc[[w for w in P.index if w.startswith("PAIR")]]
print("\n=== (b) independent competitors: payoff per (X1, X3) pair, flat 0.828 / ramp 0.70->0.828 ===")
print(pair[["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828", "flat net 0.989"]].round(1).to_string())
def pair_val(a, b, act):
    lo, hi = min(a, b), max(a, b)
    return pair.loc[f"PAIR {lo:.2f}/{hi:.2f}", act]
dists = {"uniform on 7 nets": {x: 1 / 7 for x in G},
         "round-number heavy (0.99:40%,0.90:20%,0.85:10%,0.83:5%,0.80:10%,0.75:10%,0.70:5%)": dict(zip(G, [0.05, 0.10, 0.10, 0.05, 0.10, 0.20, 0.40])),
         "mostly high (0.99:60%,0.90:25%,0.85:10%,0.80:5%)": dict(zip(G, [0, 0, 0.05, 0, 0.10, 0.25, 0.60])),
         "break-even-minded (0.75:25%,0.80:30%,0.83:20%,0.85:15%,0.90:10%)": dict(zip(G, [0, 0.25, 0.30, 0.20, 0.15, 0.10, 0]))}
print("\n=== (b) expected payoff when each competitor draws its net independently from a distribution (vs the same distribution applied to symmetric worlds only) ===")
for name, q in dists.items():
    for act in ["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828"]:
        indep = sum(q[a] * q[b] * pair_val(a, b, act) for a in G for b in G)
        symm = sum(q[a] * pair_val(a, a, act) for a in G)
        print(f"  {name[:70]:70s} {act:18s} independent {indep:7.1f} | symmetric-only {symm:7.1f}")
    q_under = sum(v for x, v in q.items() if x < 0.848); print(f"    P(one competitor's net < 0.848) = {q_under:.2f}; P(at least one of two) = {1 - (1 - q_under) ** 2:.2f}")
