"""Completeness critic, check 2: fine grid of the competitors' last-resort rate X (knife edges at our net -/+ 0.02),
competitor thresholds other than ours, competitor ramps, sex-using competitors, and realized-profit ranks."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("C:/Users/berna/Documents/introduction-to-finance/project")
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment
from lending_game.simulator import LOAN, PREF, US, draw_types, simulate, FastEvaluator
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
OUT = ROOT / "output/round1"
past = pd.read_csv(ROOT / "PastLoans.csv"); y = past["default"].to_numpy(); n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
zs = np.load(OUT / "oos_probas_sex1.npz", allow_pickle=True); p1s, p3s, pFs = zs["p1"], zs["p3"], zs["pF"]
seg = income_segment(past["income"])
rule = strategy["rule"]; tau = np.array([rule["tau"][s] for s in seg]); m = np.array([rule["margins"][s] for s in seg])
be = lambda p: p / (1 - p)
band = p2 > tau
def ramp_frac(p, s=0.25): return np.clip((p - tau) / np.maximum(s - tau, 1e-9), 0, 1)
def ours(bandrate): return np.where(band, bandrate, be(p2) + m)
ACT = {"flat net 0.828": ours(np.full(n, 0.828)), "flat net 0.989": ours(np.full(n, 0.989)), "flat net 0.700": ours(np.full(n, 0.70)),
       "ramp 0.70->0.828": ours(0.70 + 0.128 * ramp_frac(p2)), "ramp 0.60->0.828": ours(0.60 + 0.228 * ramp_frac(p2)),
       "ramp 0.50->0.828": ours(0.50 + 0.328 * ramp_frac(p2)), "model + 0.20": ours(np.minimum(be(p2) + 0.20, 1.0))}
def clone(p, X, t=None, mm=None):
    tt = tau if t is None else t; mmm = m if mm is None else mm
    return np.where(p <= tt, be(p) + mmm, X)
cases = {}
for X in np.round(np.arange(0.60, 1.00, 0.01), 2):
    cases[f"both: clone net {X:.2f}"] = (clone(p1, X), clone(p3, X))
for t in [0.15, 0.20]:
    for X in [0.75, 0.80, 0.85, 0.90, 0.99]:
        cases[f"both: +3% to {t:.2f}, net {X:.2f} above"] = (clone(p1, X, t, 0.03), clone(p3, X, t, 0.03))
cases["clone net 0.80 / clone net 0.90"] = (clone(p1, 0.80), clone(p3, 0.90))
cases["clone net 0.75 / clone net 0.99"] = (clone(p1, 0.75), clone(p3, 0.99))
cases["clone net 0.80 / clone net 0.85"] = (clone(p1, 0.80), clone(p3, 0.85))
for L, H in [(0.60, 0.828), (0.70, 0.90), (0.75, 0.99), (0.80, 0.99), (0.60, 0.99), (0.65, 0.85)]:
    cases[f"both: ramp {L:.2f}->{H:.3f}"] = (np.where(p1 <= tau, be(p1) + m, L + (H - L) * ramp_frac(p1)), np.where(p3 <= tau, be(p3) + m, L + (H - L) * ramp_frac(p3)))
# sex-using competitors (their PD computed with sex; we do not use sex)
cases["SEX both: reject above own PD 0.25"] = (np.where(p1s <= 0.25, be(p1s) + 0.03, np.inf), np.where(p3s <= 0.25, be(p3s) + 0.03, np.inf))
cases["SEX both: clone net 0.90"] = (clone(p1s, 0.90), clone(p3s, 0.90))
cases["SEX both: model + 0.20"] = (np.where(p1s <= tau, be(p1s) + m, np.minimum(be(p1s) + 0.20, 1)), np.where(p3s <= tau, be(p3s) + m, np.minimum(be(p3s) + 0.20, 1)))
cases["SEX both: everyone at +0.03"] = (np.minimum(be(p1s) + 0.03, 1), np.minimum(be(p3s) + 0.03, 1))
cases["NOSEX both: reject above own PD 0.25"] = (np.where(p1 <= 0.25, be(p1) + 0.03, np.inf), np.where(p3 <= 0.25, be(p3) + 0.03, np.inf))
cases["NOSEX both: everyone at +0.03"] = (np.minimum(be(p1) + 0.03, 1), np.minimum(be(p3) + 0.03, 1))
seeds = [0, 1, 2]
ev = FastEvaluator(cases, n, seeds, pF)
P = pd.DataFrame({a: {w: float(np.mean(v)) / 1e6 for w, v in ev.profits(r2).items()} for a, r2 in ACT.items()}).loc[list(cases)]
# realized and sex-true-PD variants
def realized(r2, cases_sub, pf=None):
    out = {}
    for seed in seeds:
        rng = np.random.default_rng(seed); types = draw_types(n, seed)
        for name, (r1, r3) in cases_sub.items():
            eff1 = r1 - PREF * (types == 0) + rng.uniform(0, 1e-9, n); eff3 = r3 - PREF * (types == 2) + rng.uniform(0, 1e-9, n)
            comp = np.minimum(eff1, eff3); eff2 = np.where(np.isfinite(r2), r2 - PREF * (types == US), np.inf)
            won = eff2 < comp
            prof = np.where(y[won] == 1, -LOAN, LOAN * r2[won]).sum() if pf is None else (LOAN * ((1 - pf[won]) * r2[won] - pf[won])).sum()
            out.setdefault(name, []).append(prof / 1e6)
    return {k: float(np.mean(v)) for k, v in out.items()}
print("=== 1. Fine grid: both competitors = clone net X (our thresholds). Expected profit (M), pF ===")
fine = P.loc[[w for w in P.index if w.startswith("both: clone net")]]
print(fine.round(2).to_string())
Xs = np.array([float(w.split()[-1]) for w in fine.index])
for lo, hi in [(0.70, 0.99), (0.70, 0.90), (0.75, 0.90), (0.78, 0.88), (0.60, 0.99)]:
    sel = (Xs >= lo - 1e-9) & (Xs <= hi + 1e-9)
    print(f"  X ~ Uniform[{lo:.2f},{hi:.2f}] (step 0.01, {sel.sum()} pts): " + " | ".join(f"{a}: {fine.loc[sel, a].mean():.1f}" for a in ["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828", "flat net 0.700", "model + 0.20"]))
grid7 = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
sel = np.isin(np.round(Xs, 2), grid7)
print(f"  matrix grid {grid7}: " + " | ".join(f"{a}: {fine.loc[sel, a].mean():.1f}" for a in ["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828", "flat net 0.700", "model + 0.20"]))
print("\n=== 2. Other thresholds, asymmetric nets, competitor ramps, sex (expected, pF) ===")
print(P.loc[[w for w in P.index if not w.startswith("both: clone net")]].round(2).to_string())
sex_cases = {k: v for k, v in cases.items() if k.startswith("SEX") or k.startswith("NOSEX")}
print("\n=== 3. Sex worlds: expected with pF (no sex), with pFs (sex-aware full-info PD), and realized ===")
rows = []
for a in ["flat net 0.828", "ramp 0.70->0.828", "model + 0.20"]:
    e0 = realized(ACT[a], sex_cases, pF); e1 = realized(ACT[a], sex_cases, pFs); rr = realized(ACT[a], sex_cases)
    for w in sex_cases:
        rows.append({"world": w, "action": a, "exp pF": e0[w], "exp pFs": e1[w], "realized": rr[w]})
print(pd.DataFrame(rows).pivot(index="world", columns="action", values=["exp pF", "exp pFs", "realized"]).round(2).to_string())
print("\n=== 4. Realized profit of all three lenders and our rank (1=best), key worlds, flat vs ramp, 3 seeds ===")
rank_cases = {k: v for k, v in cases.items() if k in ["both: clone net 0.70", "both: clone net 0.75", "both: clone net 0.80", "both: clone net 0.83", "both: clone net 0.85", "both: clone net 0.90",
                                                        "both: ramp 0.60->0.828", "both: ramp 0.70->0.900", "clone net 0.80 / clone net 0.90", "both: +3% to 0.20, net 0.90 above", "NOSEX both: reject above own PD 0.25"]}
rows = []
for name, (r1, r3) in rank_cases.items():
    for a in ["flat net 0.828", "ramp 0.70->0.828"]:
        pr = np.zeros(3)
        for seed in seeds:
            res, _ = simulate(np.column_stack([r1, ACT[a], r3]), draw_types(n, seed), y, pF, seed)
            pr += np.array([res[k]["profit_realized"] for k in range(3)]) / 3e6
        rows.append({"world": name, "action": a, "L1": pr[0], "us": pr[1], "L3": pr[2], "our rank": int(1 + (pr[0] > pr[1]) + (pr[2] > pr[1]))})
print(pd.DataFrame(rows).round(2).to_string())
