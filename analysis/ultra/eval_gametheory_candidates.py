"""Evaluate the game-theorist / rules-lawyer candidate competitor behaviours (analysis/ultra/gametheory_candidates/)
against the same action grid as analysis/band_payoff_matrix.py, plus a few anchor worlds from that grid.
Run from the project folder:  python analysis/ultra/eval_gametheory_candidates.py
Outputs: analysis/ultra/gametheory_candidates_{payoff,best}.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import US, FastEvaluator, draw_types, simulate  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
OUT = ROOT / "output" / "round1"
CAND_DIR = ROOT / "analysis" / "ultra" / "gametheory_candidates"

past = pd.read_csv(ROOT / "PastLoans.csv"); y = past["default"].to_numpy(); n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"]); chosen = strategy["rule"]; NET = chosen["tail_rate"]; seeds = [0, 1, 2]
tau_arr = np.array([chosen["tau"][s] for s in seg]); m_arr = np.array([chosen["margins"][s] for s in seg])
be = lambda p: p / (1 - p)
below = lambda p: be(p) + m_arr
ramp_frac = lambda p, s=0.25: np.clip((p - tau_arr) / np.maximum(s - tau_arr, 1e-9), 0, 1)
w_margin = lambda M: (lambda p: np.where(p <= tau_arr, below(p), np.minimum(be(p) + M, 1.0)))
w_reject = lambda t: (lambda p: np.where(p <= t, be(p) + 0.03, np.inf))
w_net = lambda X: (lambda p: np.where(p <= tau_arr, below(p), X))

WORLDS = {}
def add_world(name, regime, f1, f3=None):
    WORLDS[name] = (regime, f1, f3 or f1)
# anchors already in the grid (validation against output/round1/band_matrix_payoff.csv)
add_world("ANCHOR both: reject above PD 0.25", "reject", w_reject(0.25))
add_world("ANCHOR both: clone net 0.80", "net", w_net(0.80))
add_world("ANCHOR both: clone net 0.85", "net", w_net(0.85))
add_world("ANCHOR both: model + 0.20 in band", "margin", w_margin(0.20))
ctx = dict(tau=tau_arr, m=m_arr, seg=seg, net=NET, n=n, be=be, below=below)
for path in sorted(CAND_DIR.glob("*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = (lambda mod: (lambda p: np.asarray(mod.rates(p, ctx), dtype=float)))(mod)
    add_world(f"both: {mod.NAME}", getattr(mod, "REGIME", "extra"), f)
    if getattr(mod, "ALSO_VS_BASE", True):
        add_world(f"{mod.NAME} / reject >25%", "mixed", f, w_reject(0.25))
cases = {name: (f1(p1), f3(p3)) for name, (regime, f1, f3) in WORLDS.items()}
regime_of = pd.Series({name: regime for name, (regime, _, _) in WORLDS.items()})
for name, (r1, r3) in cases.items():  # interface sanity: finite rates in [0, 1]
    for r in (r1, r3):
        fin = np.isfinite(r); assert r.shape == (n,) and (r[fin] >= 0).all() and (r[fin] <= 1).all(), name

ours = lambda band: np.where(p2 > tau_arr, band, below(p2))
ACTIONS = {}
for X in [0.70, 0.75, 0.80, 0.828, 0.85, 0.90, 0.95, 0.989]:
    ACTIONS[f"flat net {X:.3f}"] = ours(np.full(n, X))
for L in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]:
    for H in [0.828, 0.90, 0.989]:
        if L < H:
            ACTIONS[f"ramp {L:.2f}->{H:.3f}"] = ours(L + (H - L) * ramp_frac(p2))
for M in [0.05, 0.10, 0.15, 0.20, 0.30]:
    ACTIONS[f"model + {M:.2f}"] = ours(np.minimum(be(p2) + M, 1.0))
for M in [0.10, 0.15, 0.20]:
    for s in [0.15, 0.18]:
        ACTIONS[f"hybrid: model + {M:.2f} to PD {s:.2f}, then net 0.828"] = ours(np.where(p2 <= s, be(p2) + M, NET))
ACTIONS["no offer in band"] = ours(np.full(n, np.inf))
ACTIONS["model + 0.03 (naive)"] = ours(np.minimum(be(p2) + 0.03, 1.0))
# extra diagnostic actions (NOT in the matrix grid): stage-2 best responses to undercut worlds
for X in [0.45, 0.60]:
    ACTIONS[f"(extra) flat net {X:.2f}"] = ours(np.full(n, X))
for M, c in [(0.20, 0.47), (0.20, 0.63), (0.30, 0.63)]:
    ACTIONS[f"(extra) model + {M:.2f} capped {c:.2f}"] = ours(np.minimum(be(p2) + M, c))

ev = FastEvaluator(cases, n, seeds, pF)
P = pd.DataFrame({a: {w: float(np.mean(v)) / 1e6 for w, v in ev.profits(r2).items()} for a, r2 in ACTIONS.items()}).loc[list(WORLDS)]
grid_cols = [a for a in ACTIONS if not a.startswith("(extra)")]
shortlist = ["flat net 0.700", "flat net 0.800", "flat net 0.828", "flat net 0.900", "flat net 0.989", "ramp 0.70->0.828", "ramp 0.50->0.828",
             "ramp 0.40->0.828", "ramp 0.20->0.828", "ramp 0.20->0.989", "model + 0.10", "model + 0.15", "model + 0.20", "model + 0.30",
             "hybrid: model + 0.15 to PD 0.15, then net 0.828", "hybrid: model + 0.20 to PD 0.18, then net 0.828", "no offer in band"]
best = pd.DataFrame({"regime": regime_of, "best grid action": P[grid_cols].idxmax(axis=1), "best grid payoff": P[grid_cols].max(axis=1),
                     "best incl. extra": P.idxmax(axis=1), "best incl. extra payoff": P.max(axis=1),
                     "flat net 0.828": P["flat net 0.828"], "ramp 0.70->0.828": P["ramp 0.70->0.828"],
                     "model + 0.20": P["model + 0.20"], "regret of flat 0.828 (grid)": P[grid_cols].max(axis=1) - P["flat net 0.828"],
                     "regret of ramp (grid)": P[grid_cols].max(axis=1) - P["ramp 0.70->0.828"]})
rows = {}
for name, (regime, f1, f3) in WORLDS.items():
    if not name.startswith("both") and not name.startswith("ANCHOR both"):
        continue
    r1, r3 = cases[name]
    res, _ = simulate(np.column_stack([r1, ACTIONS["flat net 0.828"], r3]), draw_types(n, 0), y, pF, 0)
    band1 = r1 != below(p1)
    rows[name] = {"lender 1 profit": res[0]["profit_realized"] / 1e6, "lender 3 profit": res[2]["profit_realized"] / 1e6,
                  "our profit": res[US]["profit_realized"] / 1e6, "our share": res[US]["share"],
                  "L1 band price mean": float(np.mean(r1[band1 & np.isfinite(r1)])) if (band1 & np.isfinite(r1)).any() else np.nan}
C = pd.DataFrame(rows).T
P.to_csv(ROOT / "analysis" / "ultra" / "gametheory_candidates_payoff.csv"); best.to_csv(ROOT / "analysis" / "ultra" / "gametheory_candidates_best.csv")
print(f"{len(WORLDS)} worlds, {len(ACTIONS)} actions ({len(grid_cols)} from the matrix grid)\n")
print("=== A. Payoff (M EUR, expected, mean of 3 type draws) of shortlisted actions per world ===")
print(P[shortlist].round(1).to_string())
print("\n=== B. Best response per world ===")
print(best.round(2).to_string())
print("\n=== C. Competitor realized profit when we play flat net 0.828 (symmetric worlds) ===")
print(C.round(2).to_string())
