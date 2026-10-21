"""Student lens: the slides' formula taken literally, R = f(X) + M with a 'conservative' M = 10%, everyone served."""
import numpy as np
NAME = "lecture formula p + 10% to everyone"
REGIME = "cheap"
ALSO_VS_BASE = True

def rates(p, ctx):
    return np.minimum(np.asarray(p, float) + 0.10, 1.0)
