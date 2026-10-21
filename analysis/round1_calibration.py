"""Round-1 post-mortem without the diagnostic file: which rival profiles reproduce our league-table outcome
(market share 51%, P&L 28.7 M EUR with the submitted rule), and how would alternative rules have done against them?
Candidates: higher safe-part margins, steeper band schedules (adjusted break-even + larger X), flat nets in the band,
and combinations.  Exact applicant types (1/3 each), fair ties, profits with the diagnostic full-information PD.
Run from the project folder:  python analysis/round1_calibration.py [n_draws]
Output: output/round1/round1_calibration.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
n, pF, p1, p2, p3 = bpm.n, bpm.pF, bpm.p1, bpm.p2, bpm.p3
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40
CAP = 0.989
ACTUAL_SHARE, ACTUAL_PNL = 0.51, 28.68
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
in_band = p2 > tau_arr
w = 1 / (1 + np.exp(-(p2 - tau_arr - 0.01) / 0.01))


def rule(extra_safe=0.0, band_kind="adj", band_level=0.05):
    safe = be(p2) + m_arr + extra_safe
    if band_kind == "adj":
        band = np.minimum(adj + band_level, CAP)
    elif band_kind == "flat":
        band = np.full(n, min(band_level, CAP))
    elif band_kind == "ramp":  # linear from band_level[0] at the threshold to band_level[1] at 25% PD
        frac = np.clip((p2 - tau_arr) / np.maximum(0.25 - tau_arr, 1e-9), 0, 1)
        band = np.minimum(band_level[0] + (band_level[1] - band_level[0]) * frac, CAP)
    return np.round(np.minimum(safe + w * (band - safe), CAP), 4)


CANDS = {
    "current rule (safe m, band adj + 0.05)": rule(),
    "safe +2 pts": rule(0.02),
    "safe +4 pts": rule(0.04),
    "safe +6 pts": rule(0.06),
    "band adj + 0.10": rule(0, "adj", 0.10),
    "band adj + 0.20": rule(0, "adj", 0.20),
    "band ramp 0.45 -> 0.65": rule(0, "ramp", (0.45, 0.65)),
    "band flat 0.70": rule(0, "flat", 0.70),
    "band flat 0.85": rule(0, "flat", 0.85),
    "band flat 0.989 (net)": rule(0, "flat", 0.989),
    "safe +4 & band adj + 0.15": rule(0.04, "adj", 0.15),
    "safe +4 & band flat 0.85": rule(0.04, "flat", 0.85),
    "safe +6 & band flat 0.989": rule(0.06, "flat", 0.989),
}


def evaluate(r1, r2s, r3):
    """Our expected profit (M), market share, and the safe/band split, exact types and fair ties."""
    out = {}
    for a, r2 in r2s.items():
        wp = np.zeros(n)
        for t in range(3):
            E = np.column_stack([r1 - PREF * (t == 0), r2 - PREF * (t == 1), r3 - PREF * (t == 2)])
            m = E.min(axis=1)
            tie = (E <= m[:, None] + 1e-12) & np.isfinite(m)[:, None]
            wp += np.where(tie[:, 1], 1.0 / np.maximum(tie.sum(axis=1), 1), 0.0) / 3
        prof = wp * LOAN * ((1 - pF) * r2 - pF)
        out[a] = (prof.sum() / 1e6, wp.mean(), prof[~in_band].sum() / 1e6, prof[in_band].sum() / 1e6, wp[~in_band].mean(), wp[in_band].mean())
    return out


records = []
for profile in rivals.MIXTURES:
    W = list(rivals.MIXTURES[profile].values())[0]
    rng = np.random.default_rng(8080)
    print(f"{profile}: {N_DRAWS} draws...", flush=True)
    for d in range(N_DRAWS):
        r1 = rivals.draw_rival(rng, p1, W, profile, tau_arr, n)[0]
        r3 = rivals.draw_rival(rng, p3, W, profile, tau_arr, n)[0]
        for a, (pr, sh, ps, pb, ss, sb) in evaluate(r1, CANDS, r3).items():
            records.append({"profile": profile, "action": a, "profit": pr, "share": sh, "profit safe": ps, "profit band": pb, "share safe": ss, "share band": sb})
D = pd.DataFrame(records)
M = D.groupby(["action", "profile"]).mean(numeric_only=True)
cur = M.loc["current rule (safe m, band adj + 0.05)"]
print("\n=== Our submitted rule, by rival profile: simulated share and profit vs the actual 51% / 28.7 M ===")
cur_show = cur.copy()
cur_show["distance to actual"] = np.sqrt(((cur_show["share"] - ACTUAL_SHARE) / 0.05) ** 2 + ((cur_show["profit"] - ACTUAL_PNL) / 5) ** 2)
cur_show = cur_show.sort_values("distance to actual")
print(pd.DataFrame({"share %": cur_show["share"] * 100, "profit M": cur_show["profit"], "safe-part profit M": cur_show["profit safe"], "band profit M": cur_show["profit band"],
                    "share safe %": cur_show["share safe"] * 100, "share band %": cur_show["share band"] * 100, "distance": cur_show["distance to actual"]}).round(1).to_string())
consistent = list(cur_show.index[:3])
print(f"\nProfiles closest to the actual outcome: {consistent}")

print("\n=== Candidate rules: profit (M) by profile, then the average over the closest profiles and the worst profile ===")
P = M["profit"].unstack("profile").loc[list(CANDS)]
S = M["share"].unstack("profile").loc[list(CANDS)]
P["avg, consistent profiles"] = P[consistent].mean(axis=1)
P["worst profile"] = P[list(rivals.MIXTURES)].min(axis=1)
P["share, consistent %"] = S[consistent].mean(axis=1) * 100
print(P.round(1).to_string())
pd.concat({"profit": M["profit"].unstack("profile"), "share": M["share"].unstack("profile")}, axis=1).to_csv(bpm.OUT / "round1_calibration.csv")
