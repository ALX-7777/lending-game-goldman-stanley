"""Student lens: timid lender of last resort. Break-even + 3% up to PD 0.20, flat 0.70 above."""
import numpy as np
NAME = "flat 0.70 above PD 0.20"
REGIME = "net"
ALSO_VS_BASE = True
X, T = 0.70, 0.20

def rates(p, ctx):
    p = np.asarray(p, float)
    return np.where(p <= T, np.minimum(p / (1 - p) + 0.03, 1.0), X)
