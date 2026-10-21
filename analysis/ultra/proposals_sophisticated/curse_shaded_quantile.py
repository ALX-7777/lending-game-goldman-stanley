"""Curse-shaded pricing: a team that measures the hidden information per income segment (std of
p_full - p_own, computable from PastLoans since all three signals are there) and prices every band
applicant as if its PD were z standard deviations worse ('bid as if you held the worst signal').
Band prices land between ~0.25 (10-20k, PD 0.09) and ~0.70 (income 0, PD 0.30)."""
import numpy as np

NAME = "curse-shaded z=1.5 (segment sigma)"
REGIME = "margin"
ALSO_VS_BASE = True
SIGMA = {"0": 0.071, "1-5k": 0.050, "5-10k": 0.052, "10-20k": 0.025, "20-40k": 0.008, "40k+": 0.003}
Z = 1.5


def rates(p, ctx):
    sig = np.array([SIGMA[s] for s in ctx["seg"]])
    p_shaded = np.minimum(p + Z * sig, 0.60)
    in_band = np.minimum(ctx["be"](p_shaded) + ctx["m"] + 0.05, 1.0)
    return np.where(p <= ctx["tau"], ctx["below"](p), in_band)
