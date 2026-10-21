"""Scratch: realized default rate of the loans 'model + 0.15' wins inside the band in world 'both: reject above PD 0.25'."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment
from lending_game.simulator import US, draw_types, simulate

OUT = ROOT / "output" / "round1"
past = pd.read_csv(ROOT / "PastLoans.csv"); y = past["default"].to_numpy(); n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json")); chosen = strategy["rule"]
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
tau_arr = np.array([chosen["tau"][s] for s in seg]); m_arr = np.array([chosen["margins"][s] for s in seg])
be = lambda p: p / (1 - p)
below = lambda p: be(p) + m_arr
w_reject = lambda t: (lambda p: np.where(p <= t, be(p) + 0.03, np.inf))
ours = lambda band: np.where(p2 > tau_arr, band, below(p2))

band = p2 > tau_arr
print(f"n={n}, band size={band.sum()} ({band.mean():.3f}), band mean pF={pF[band].mean():.4f}, band realized default={y[band].mean():.4f}")
r1, r3 = w_reject(0.25)(p1), w_reject(0.25)(p3)
actions = {"model + 0.15": ours(np.minimum(be(p2) + 0.15, 1.0)), "flat net 0.828": ours(np.full(n, 0.828)),
           "model + 0.20": ours(np.minimum(be(p2) + 0.20, 1.0)), "model + 0.10": ours(np.minimum(be(p2) + 0.10, 1.0))}
for name, r2 in actions.items():
    for seed in [0, 1, 2]:
        res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
        won = choice == US
        wb = won & band
        wb_comp_served = wb & (np.isfinite(r1) | np.isfinite(r3))
        wb_abandoned = wb & ~(np.isfinite(r1) | np.isfinite(r3))
        print(f"{name:16s} seed={seed} total realized={res[US]['profit_realized']/1e6:8.3f}M expected={res[US]['profit_expected']/1e6:8.3f}M | "
              f"won in band={wb.sum():5d} realized default={y[wb].mean():.4f} expected pF={pF[wb].mean():.4f} mean rate={r2[wb].mean():.4f} "
              f"band realized profit={np.where(y[wb]==1,-1e4,1e4*r2[wb]).sum()/1e6:7.3f}M | of which abandoned-by-both={wb_abandoned.sum()} (def {y[wb_abandoned].mean() if wb_abandoned.any() else float('nan'):.4f}), "
              f"contested={wb_comp_served.sum()} (def {y[wb_comp_served].mean() if wb_comp_served.any() else float('nan'):.4f})")
