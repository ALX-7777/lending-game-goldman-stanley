"""Competitor archetype: the base behaviour (break-even + 3%, no offer above PD 0.25) with a realistic model error.

The competitor's PD estimate is its own full-information-free estimate plus uninformative noise on the LOGIT scale
(sd S = 0.25), i.e. a quick logistic regression or an sklearn-default (overfit) GBM instead of a tuned boosted tree.
Calibration: a 5-fold out-of-sample logistic regression on X + signal deviates from the boosted-tree PD by sd 0.235 logit
inside the risky band (overfit GBM: 0.235; AUC 0.728 vs 0.735). Then: rate = q/(1-q) + 0.03 if q <= 0.25, else no offer.

Why the noise matters: errors at the 25% threshold enlarge the pool rejected by BOTH competitors (3.2% of the band vs 1.8%
noiseless) and make it less cursed (realized default of what our 0.828 net wins: 35% vs 43%), so the safety-net rent rises.

Determinism: the seed is derived from the PD array itself so that lender 1 (p1) and lender 3 (p3) draw INDEPENDENT noise
while each lender's draw is reproducible across calls. A fixed seed shared by both would make their errors identical and
shrink the pool rejected by both. Identical to analysis/ultra/lens_student/student_noisy_lr.py.
"""
import numpy as np

NAME = "noisy PD (logit sd 0.25), +3%, reject >25%"
REGIME = "cheap"
ALSO_VS_BASE = True
S = 0.25  # sd of the logit noise added to the competitor's PD estimate


def rates(p, ctx):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    rng = np.random.default_rng(int(round(p.sum() * 1e6)) % (2**31 - 1))  # deterministic, differs for lender 1 and 3
    q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + S * rng.standard_normal(p.shape))))
    return np.where(q <= 0.25, np.minimum(q / (1 - q) + 0.03, 1.0), np.inf)
