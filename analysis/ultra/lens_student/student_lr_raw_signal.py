"""Student lens: logistic regression with the raw signal (signal == 0 treated as a value, not as 'no signal').
The 21% with signal 0 get +0.26 logit (PD 0.095 -> 0.12), the others -0.08, plus sd 0.20 noise; then +3%, reject >25%."""
import numpy as np
NAME = "LR with raw signal (21% overpriced +0.26 logit), +3%, reject >25%"
REGIME = "cheap"
ALSO_VS_BASE = True

def rates(p, ctx):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    rng = np.random.default_rng(int(round(p.sum() * 1e6)) % (2**31 - 1))
    zero = rng.random(p.shape) < 0.21                 # applicants whose signal is exactly 0 (uninformative)
    lg = np.log(p / (1 - p)) + np.where(zero, 0.26, -0.08) + 0.20 * rng.standard_normal(p.shape)
    q = 1 / (1 + np.exp(-lg))
    return np.where(q <= 0.25, np.minimum(q / (1 - q) + 0.03, 1.0), np.inf)
