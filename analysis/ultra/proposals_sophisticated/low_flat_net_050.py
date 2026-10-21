"""Low flat net: a team that reasons 'my band has PD ~0.16 so break-even is ~0.20; 0.50 is 2.5x that
and undercuts any safety net the others might put' without computing the abandoned-pool default."""
import numpy as np

NAME = "clone net 0.50 (low)"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    return np.where(p <= ctx["tau"], ctx["below"](p), 0.50)
