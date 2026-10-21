"""Student lens: timid clone net. Break-even + 3% up to PD 0.12, flat 0.60 above (a clone net below 0.70)."""
import numpy as np
NAME = "flat 0.60 above PD 0.12 (timid clone net)"
REGIME = "net"
ALSO_VS_BASE = True
X, T = 0.60, 0.12

def rates(p, ctx):
    p = np.asarray(p, float)
    return np.where(p <= T, np.minimum(p / (1 - p) + 0.03, 1.0), X)
