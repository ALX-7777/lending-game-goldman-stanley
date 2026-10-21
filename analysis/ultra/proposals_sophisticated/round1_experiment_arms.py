"""Round-1 price experiment: a data-science team uses round 1 to LEARN the market, randomly assigning
band applicants to four price arms (model+0.05, model+0.15, model+0.30, net 0.95) so that the round-1
results reveal the demand curve for round 2. Seed derived from the PD array: independent arms per lender."""
import numpy as np

NAME = "round-1 experiment arms (+.05/+.15/+.30/net .95)"
REGIME = "mixed"
ALSO_VS_BASE = True


def rates(p, ctx):
    rng = np.random.default_rng(12345 + int(np.abs(p).sum() * 1e6) % (2**31))
    arm = rng.integers(0, 4, len(p))
    be = ctx["be"](p)
    band = np.select([arm == 0, arm == 1, arm == 2], [be + 0.05, be + 0.15, be + 0.30], default=0.95)
    return np.where(p <= ctx["tau"], ctx["below"](p), np.minimum(band, 1.0))
