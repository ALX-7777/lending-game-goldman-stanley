"""Randomized safety net: a team that sees the net as a Bertrand undercutting game with no pure
equilibrium and plays a mixed strategy, drawing each band applicant's net from U(0.60, 0.99).
The seed is derived from the PD array so lender 1 and lender 3 get independent (but reproducible) draws."""
import numpy as np

NAME = "randomized net U(0.60,0.99)"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    rng = np.random.default_rng(12345 + int(np.abs(p).sum() * 1e6) % (2**31))
    net = rng.uniform(0.60, 0.99, len(p))
    return np.where(p <= ctx["tau"], ctx["below"](p), net)
