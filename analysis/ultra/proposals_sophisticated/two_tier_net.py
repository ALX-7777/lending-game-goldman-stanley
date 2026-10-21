"""Two-tier safety net: 'moderately risky people that others abandon still have PD ~0.15, so a 0.45
net is both an undercut of any competitor net and 2x their break-even; above PD 0.20 charge the cap'."""
import numpy as np

NAME = "two-tier net 0.45 (PD<=0.20) / 0.95"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    tier = np.where(p <= 0.20, 0.45, 0.95)
    return np.where(p <= ctx["tau"], ctx["below"](p), tier)
