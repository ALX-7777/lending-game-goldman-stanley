"""Segment-gated band policy: a team that knows the curse is concentrated at low income serves the
band at model + 0.08 where hidden information is small (income >= 5k) and only nets (0.90) where it
is large (income 0 and 1-5k)."""
import numpy as np

NAME = "segment-gated: +0.08 if income>=5k, net 0.90 below"
REGIME = "mixed"
ALSO_VS_BASE = True


def rates(p, ctx):
    low_income = np.isin(ctx["seg"], ["0", "1-5k"])
    band = np.where(low_income, 0.90, np.minimum(ctx["be"](p) + 0.08, 1.0))
    return np.where(p <= ctx["tau"], ctx["below"](p), band)
