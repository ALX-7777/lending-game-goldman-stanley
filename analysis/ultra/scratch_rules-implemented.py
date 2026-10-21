"""Scratch: check that lending_game/simulator.py implements the game rules, and FastEvaluator == simulate."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"C:/Users/berna/Documents/introduction-to-finance/project")
sys.path.insert(0, str(ROOT))
from lending_game.simulator import LOAN, PREF, US, FastEvaluator, draw_types, simulate, effective_rates

past = pd.read_csv(ROOT / "PastLoans.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(ROOT / "output/round1/strategy_round1.json"))
z = np.load(ROOT / f"output/round1/oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
be = lambda p: p / (1 - p)

# ---------- 1. reference implementation of the written rules (explicit per-type formula, python loop on a sample)
def reference(rates, types, y, pF):
    n = len(types)
    choice = np.full(n, -1)
    prof_real = np.zeros(3); prof_exp = np.zeros(3); loans = np.zeros(3, int)
    for i in range(n):
        r1, r2, r3 = rates[i]
        k = types[i]  # 0,1,2 <-> lender 1,2,3
        cand = [r1 - (0.02 if k == 0 else 0.0), r2 - (0.02 if k == 1 else 0.0), r3 - (0.02 if k == 2 else 0.0)]
        if all(np.isinf(c) for c in cand):
            continue
        j = int(np.argmin(cand))
        choice[i] = j
        loans[j] += 1
        prof_real[j] += -LOAN if y[i] == 1 else LOAN * rates[i, j]
        prof_exp[j] += LOAN * ((1 - pF[i]) * rates[i, j] - pF[i])
    return choice, prof_real, prof_exp, loans

print("=== 1. worked example from the instructions: r=(0.05,0.06,0.08) ===")
ex = np.array([[0.05, 0.06, 0.08]] * 3)
res, ch = simulate(ex, np.array([0, 1, 2]), np.zeros(3, int), np.zeros(3), 0)
print("choices by type 1,2,3 (0-based lender idx):", ch.tolist(), " expected [0, 1, 0]")

print("\n=== 2. random rate matrices with inf, vs reference loop (sample of 5000 applicants) ===")
rng = np.random.default_rng(7)
m = 5000
idx = rng.choice(n, m, replace=False)
for trial in range(3):
    R = rng.uniform(0.0, 1.0, (m, 3))
    R[rng.uniform(size=(m, 3)) < 0.3] = np.inf      # 30% no-offer cells
    types = draw_types(m, trial)
    res, ch = simulate(R, types, y[idx], pF[idx], trial)
    rch, rreal, rexp, rloans = reference(R, types, y[idx], pF[idx])
    sim_real = np.array([r["profit_realized"] for r in res]); sim_exp = np.array([r["profit_expected"] for r in res])
    sim_loans = np.array([r["loans"] for r in res])
    print(f"trial {trial}: choices identical={np.array_equal(ch, rch)} (mismatch {int((ch != rch).sum())}/{m}); "
          f"no-loan count sim={int((ch == -1).sum())} ref={int((rch == -1).sum())} all-inf rows={int(np.isinf(R).all(axis=1).sum())}; "
          f"loans sim={sim_loans.tolist()} ref={rloans.tolist()}; "
          f"max|profit_realized diff|={np.abs(sim_real - rreal).max():.6g}; max|profit_expected diff|={np.abs(sim_exp - rexp).max():.6g}")
    has_offer = np.isfinite(R).any(axis=1)
    print(f"   has_offer & no loan: {int((has_offer & (ch == -1)).sum())};  no offer & loan: {int((~has_offer & (ch != -1)).sum())}")
    won = ch == US
    manual = np.where(y[idx][won] == 1, -LOAN, LOAN * R[won, US]).sum()
    print(f"   lender2 realized: sim={sim_real[US]:.2f} manual={manual:.2f}; defaults among won={int(y[idx][won].sum())}, won={int(won.sum())}")

print("\n=== 3. FastEvaluator.profits(r2) vs simulate()['profit_expected'] for lender 2, full data n=%d ===" % n)
from lending_game.features import income_segment
seg = income_segment(past["income"])
chosen = strategy["rule"]; NET = chosen["tail_rate"]
tau_arr = np.array([chosen["tau"][s] for s in seg]); m_arr = np.array([chosen["margins"][s] for s in seg])
below = lambda p: be(p) + m_arr
rA1 = np.where(p1 <= tau_arr, below(p1), np.minimum(be(p1) + 0.10, 1.0))          # model + 0.10 in band (lender 1)
rA3 = np.where(p3 <= 0.25, be(p3) + 0.03, np.inf)                                 # reject above 0.25 (lender 3)
rB1 = np.where(p1 <= tau_arr, below(p1), 0.90)                                     # clone net 0.90
rB3 = np.where(p3 <= tau_arr, below(p3), np.inf)                                   # reject band
rng2 = np.random.default_rng(12345)
rC1 = np.where(rng2.uniform(size=n) < 0.2, np.inf, rng2.uniform(0, 1, n))         # random
rC3 = np.where(rng2.uniform(size=n) < 0.2, np.inf, rng2.uniform(0, 1, n))
comp = {"A: model+0.10 / reject>25%": (rA1, rA3), "B: clone net 0.90 / reject band": (rB1, rB3), "C: random with 20% inf": (rC1, rC3)}
ours = {
    "flat net 0.828": np.where(p2 <= tau_arr, below(p2), NET),
    "ramp 0.70->0.828": np.where(p2 <= tau_arr, below(p2), 0.70 + (NET - 0.70) * np.clip((p2 - tau_arr) / np.maximum(0.25 - tau_arr, 1e-9), 0, 1)),
    "model + 0.15": np.where(p2 <= tau_arr, below(p2), np.minimum(be(p2) + 0.15, 1.0)),
    "no offer in band": np.where(p2 <= tau_arr, below(p2), np.inf),
}
seeds = [0, 1, 2]
ev = FastEvaluator(comp, n, seeds, pF)
maxabs = 0.0; maxrel = 0.0
for aname, r2 in ours.items():
    fe = ev.profits(r2)
    for cname, (r1, r3) in comp.items():
        for si, seed in enumerate(seeds):
            types = draw_types(n, seed)
            res, ch = simulate(np.column_stack([r1, r2, r3]), types, y, pF, seed)
            s = res[US]["profit_expected"]; f = fe[cname][si]
            eff = np.column_stack([r1, r2, r3]).astype(float); eff[np.arange(n), types] -= PREF
            cm = np.minimum(eff[:, 0], eff[:, 2])
            ties = int(((eff[:, 1] == cm) & np.isfinite(eff[:, 1])).sum())
            d = f - s; maxabs = max(maxabs, abs(d)); maxrel = max(maxrel, abs(d) / max(abs(s), 1))
            print(f"{aname:18s} | {cname:32s} | seed {seed}: simulate={s/1e6:10.4f} M  fast={f/1e6:10.4f} M  diff={d:12.4f} EUR  exact ties={ties}")
print(f"max |diff| = {maxabs:.6f} EUR, max rel diff = {maxrel:.3e}")

print("\n=== 4. tie-breaking: our rate exactly equal to a competitor's (both flat 0.828 in band) ===")
r1t = np.where(p1 <= tau_arr, below(p1), 0.828); r3t = np.where(p3 <= tau_arr, below(p3), np.inf)
r2t = ours["flat net 0.828"]
ev2 = FastEvaluator({"tie": (r1t, r3t)}, n, seeds, pF)
fe = ev2.profits(r2t)["tie"]
for si, seed in enumerate(seeds):
    types = draw_types(n, seed)
    res, ch = simulate(np.column_stack([r1t, r2t, r3t]), types, y, pF, seed)
    eff = np.column_stack([r1t, r2t, r3t]).astype(float); eff[np.arange(n), types] -= PREF
    ties = int(((eff[:, 1] == eff[:, 0]) & np.isfinite(eff[:, 1])).sum())
    print(f"seed {seed}: simulate={res[US]['profit_expected']/1e6:.4f} M  fast={fe[si]/1e6:.4f} M  diff={(fe[si]-res[US]['profit_expected'])/1e6:.4f} M  exact ties={ties}  (sim loans={res[US]['loans']})")

print("\n=== 5. edge cases: cap at 100% and NaN ===")
R = np.array([[1.5, np.inf, np.inf], [np.nan, 0.10, 0.20], [0.5, 0.5, np.inf]])
res, ch = simulate(R, np.array([1, 1, 2]), np.zeros(3, int), np.zeros(3), 0)
print("rate 1.5 (> cap) accepted as a loan? choice row0 =", ch[0], "(simulator does not enforce the 100% cap)")
print("NaN as 'no offer' for lender 1 while lender 2 offers 0.10: choice row1 =", ch[1], "(1 if NaN meant no offer; -1 means applicant wrongly gets no loan)")
sub = pd.read_csv(ROOT / 'output/round1/rates_round1.csv')
print("rates_round1.csv columns:", list(sub.columns), " NaN count:", int(sub['rate'].isna().sum()), " max rate:", sub['rate'].max(), " min rate:", sub['rate'].min())

print("\n=== 6. exact ties inside the real band-matrix grid (seed 0): FastEvaluator vs simulate ===")
import importlib.util
spec = importlib.util.spec_from_file_location("bpm", ROOT / "analysis/band_payoff_matrix.py")
bpm = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(bpm)
P = pd.read_csv(ROOT / "output/round1/band_matrix_payoff.csv", index_col=0)
print("payoff csv shape (worlds x actions):", P.shape, "| WORLDS in module:", len(bpm.WORLDS), "| ACTIONS:", len(bpm.ACTIONS))
types0 = draw_types(n, 0)
rows = []
for wname, (r1, r3) in bpm.cases.items():
    for aname, r2 in bpm.ACTIONS.items():
        eff = np.column_stack([r1, r2, r3]).astype(float); eff[np.arange(n), types0] -= PREF
        cm = np.minimum(eff[:, 0], eff[:, 2])
        ties = int(((eff[:, 1] == cm) & np.isfinite(eff[:, 1])).sum())
        if ties > 0:
            ev1 = FastEvaluator({wname: (r1, r3)}, n, [0], pF)
            f = ev1.profits(r2)[wname][0]
            s = simulate(np.column_stack([r1, r2, r3]), types0, y, pF, 0)[0][US]["profit_expected"]
            rows.append((wname, aname, ties, f / 1e6, s / 1e6, (f - s) / 1e6, P.loc[wname, aname] if (wname in P.index and aname in P.columns) else np.nan))
T = pd.DataFrame(rows, columns=["world", "action", "exact ties", "fast seed0 (M)", "simulate seed0 (M)", "fast-sim (M)", "csv payoff (M)"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"{len(T)} (world, action) pairs with exact ties out of {len(bpm.cases) * len(bpm.ACTIONS)}")
print(T.sort_values("fast-sim (M)", ascending=False).round(3).to_string())
