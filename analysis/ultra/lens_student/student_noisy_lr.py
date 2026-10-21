"""Student lens: quick logistic regression / overfit GBM. Own PD = ours + logit noise (sd 0.25, calibrated on a
5-fold OOS logistic regression vs the boosted tree inside the band), then break-even + 3%, no offer above PD 0.25."""
import numpy as np
NAME = "noisy PD (logit sd 0.25), +3%, reject >25%"
REGIME = "cheap"
ALSO_VS_BASE = True
S = 0.25

def rates(p, ctx):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    rng = np.random.default_rng(int(round(p.sum() * 1e6)) % (2**31 - 1))  # deterministic, differs for lender 1 and 3
    q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + S * rng.standard_normal(p.shape))))
    return np.where(q <= 0.25, np.minimum(q / (1 - q) + 0.03, 1.0), np.inf)
