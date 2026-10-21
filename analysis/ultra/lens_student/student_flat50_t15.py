"""Student lens: 'risky people pay a flat 50%'. Break-even + 3% up to PD 0.15, flat 0.50 above."""
import numpy as np
NAME = "flat 0.50 above PD 0.15"
REGIME = "margin"
ALSO_VS_BASE = True
X, T = 0.50, 0.15

def rates(p, ctx):
    p = np.asarray(p, float)
    return np.where(p <= T, np.minimum(p / (1 - p) + 0.03, 1.0), X)
