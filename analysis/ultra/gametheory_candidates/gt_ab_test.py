"""Hedger / experimenter lens: unsure whether to serve the band at a margin or net it, the team A/B tests in
round 1: a random half of band members get model + 0.10, the other half get the net 0.99. Deterministic split."""
import numpy as np

NAME = "GT A/B test: half model + 0.10, half net 0.99"
REGIME = "mixed"
ALSO_VS_BASE = True


def rates(p, ctx):
    rng = np.random.default_rng(12345)
    arm = rng.random(ctx["n"]) < 0.5
    served = np.minimum(ctx["be"](p) + 0.10, 1.0)
    band = np.where(arm, served, 0.99)
    return np.where(p <= ctx["tau"], ctx["below"](p), band)
