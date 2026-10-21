"""Student lens: pipeline bug. Rows with signal == 0 are dropped/left NaN, so 21% of applicants (uninformative
subset) get no offer; everybody else break-even + 3%, no offer above PD 0.25."""
import numpy as np
NAME = "no offer to the 21% with signal = 0; else +3%, reject >25%"
REGIME = "reject"
ALSO_VS_BASE = True

def rates(p, ctx):
    p = np.asarray(p, float)
    rng = np.random.default_rng(int(round(p.sum() * 1e6)) % (2**31 - 1))
    dropped = rng.random(p.shape) < 0.21
    r = np.where(p <= 0.25, np.minimum(p / (1 - p) + 0.03, 1.0), np.inf)
    return np.where(dropped, np.inf, r)
