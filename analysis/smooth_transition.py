"""Step or smooth transition at the threshold?

Candidate family:  rate(p) = safe(p) + w(p) * [band(p) - safe(p)],  with safe(p) = p/(1-p) + m(segment) and
w(p) = sigmoid(6 * (p - tau)/width - 3): about 0.05 at the threshold, 0.95 at threshold + width.  width -> 0 is the
step we use now.  Two band targets: the ramp 0.45 -> 0.65 and "selection-adjusted break-even + 0.25".
Evaluated against the six rival profiles on fresh draws, like cross_profiles.py.
Run from the project folder:  python analysis/smooth_transition.py [n_draws]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF, draw_types  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be, ramp_frac = bpm.tau_arr, bpm.m_arr, bpm.be, bpm.ramp_frac
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
safe = be(p2) + m_arr
in_band = p2 > tau_arr

floor_t = pd.read_csv(bpm.OUT / "breakeven_ramp_smart.csv")
mids = np.array([0.11, 0.135, 0.165, 0.20, 0.24, 0.28, 0.35])
floor_vals = floor_t["at it: selection-adjusted break-even"].to_numpy()
floor_p = np.interp(p2, mids, floor_vals)  # selection-adjusted break-even as a function of PD

targets = {"ramp 0.45->0.65": 0.45 + 0.20 * ramp_frac(p2),
           "floor + 0.25": np.minimum(floor_p + 0.25, 0.828)}


def sigmoid_w(width):
    if width == 0:
        return in_band.astype(float)
    return 1 / (1 + np.exp(-(6 * (p2 - tau_arr) / width - 3)))


CANDS = {}
for tname, band in targets.items():
    for width in [0.0, 0.01, 0.02, 0.03, 0.05]:
        w = sigmoid_w(width)
        r = safe + w * (band - safe)
        label = f"{tname}, {'step' if width == 0 else f'sigmoid width {width:.0%} PD'}"
        CANDS[label] = np.where(in_band, r, np.where(width == 0, safe, r))  # below the threshold the sigmoid is ~0 anyway
CANDS["flat net 0.828 (first submission)"] = np.where(in_band, 0.828, safe)

PROFILES = [("mixed", "baseline (45/15/15/15/10)"), ("smart", list(rivals.MIXTURES["smart"])[0]), ("clones", list(rivals.MIXTURES["clones"])[0]),
            ("modelfollowers", list(rivals.MIXTURES["modelfollowers"])[0]), ("netters", list(rivals.MIXTURES["netters"])[0]),
            ("verysmart", list(rivals.MIXTURES["verysmart"])[0])]


def evaluate(r1, r2s, r3, types):
    out = {}
    eff1 = r1 - PREF * (types == 0)
    eff3 = r3 - PREF * (types == 2)
    for a, r2 in r2s.items():
        eff2 = np.where(np.isfinite(r2), r2 - PREF * (types == 1), np.inf)
        E = np.column_stack([eff1, eff2, eff3])
        best = E.min(axis=1)
        choice = np.where(np.isfinite(best), E.argmin(axis=1), -1)
        out[a] = [(LOAN * ((1 - pF[choice == k]) * rk[choice == k] - pF[choice == k])).sum() / 1e6 for k, rk in enumerate([r1, r2, r3])]
    return out


records = []
for profile, mname in PROFILES:
    W = rivals.MIXTURES[profile][mname]
    rng = np.random.default_rng(3030)
    print(f"{profile}: {N_DRAWS} fresh draws...", flush=True)
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        types = draw_types(n, int(rng.integers(1 << 30)))
        for a, (q1, q2, q3) in evaluate(r1, CANDS, r3, types).items():
            records.append({"profile": profile, "action": a, "profit": q2, "rank": 1 + (q1 > q2) + (q3 > q2)})
D = pd.DataFrame(records)
g = D.groupby(["action", "profile"])
M = g["profit"].mean().unstack("profile")[[p for p, _ in PROFILES]].loc[list(CANDS)]
M["average"] = M.mean(axis=1)
R1 = g["rank"].apply(lambda s: (s == 1).mean()).unstack("profile")[[p for p, _ in PROFILES]].loc[list(CANDS)]
R1["average"] = R1.mean(axis=1)
print("\n=== MEAN expected profit (M EUR) ===")
print(M.round(1).to_string())
print("\n=== P(first) ===")
print(R1.round(2).to_string())
M.to_csv(bpm.OUT / "smooth_transition_mean.csv")
R1.to_csv(bpm.OUT / "smooth_transition_rank1.csv")

# how many applicants are priced in the transition zone, and at what rates
for width in [0.01, 0.02, 0.03, 0.05]:
    w = sigmoid_w(width)
    zone = in_band & (w < 0.9)
    r = (safe + w * (targets["ramp 0.45->0.65"] - safe))[zone]
    print(f"width {width:.0%}: {zone.sum():6d} band applicants in the transition zone, priced {r.min():.2f}-{r.max():.2f} (mean {r.mean():.2f})")
