"""Segment-gated band policy (competitor archetype for analysis/band_payoff_matrix.py).

A team that has noticed in PastLoans that the hidden information (the winner's curse) is
concentrated at low income (std of the rivals' signals in PD points: 0.069 at income 0,
0.049 at 1-5k, 0.050 at 5-10k, 0.025 at 10-20k, < 0.01 above) serves the risky band at a thin
curse margin where it is safe (model + 0.08 for income >= 5k) and only plays lender of last
resort where it is not (safety net LOW_NET = 0.90 at income 0 and 1-5k).

Below our thresholds (p <= ctx["tau"]) the competitor prices like the clone worlds, with our
own below-threshold schedule ctx["below"](p) = p/(1-p) + m(segment).

Identical in behaviour to analysis/ultra/proposals_sophisticated/segment_gated_band.py.
"""
import numpy as np

NAME = "segment-gated: +0.08 if income>=5k, net 0.90 below"
REGIME = "mixed"
ALSO_VS_BASE = True

LOW_NET = 0.90          # safety-net rate on the low-income band (income 0 and 1-5k)
HIGH_MARGIN = 0.08      # curse margin over the break-even rate on the band where income >= 5k
LOW_INCOME_SEGMENTS = ("0", "1-5k")


def rates(p, ctx):
    """Offered rate per applicant given the competitor's own PD estimate p (array in (0, 1)).

    Returns a float array of shape (len(p),). This archetype never withholds an offer
    (no np.inf); all rates are finite and lie in [0, 1]. Deterministic (no RNG).
    """
    p = np.asarray(p, dtype=float)
    seg = np.asarray(ctx["seg"])
    tau = np.asarray(ctx["tau"], dtype=float)

    low_income = np.isin(seg, LOW_INCOME_SEGMENTS)
    band = np.where(low_income, LOW_NET, np.minimum(ctx["be"](p) + HIGH_MARGIN, 1.0))
    out = np.where(p <= tau, ctx["below"](p), band)
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)
