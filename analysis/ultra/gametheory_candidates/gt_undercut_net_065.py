"""Game-theorist lens: milder Bertrand undercut. The team expects rival nets at 0.70-0.99 and prices the band
at a flat 0.65, just under the lowest round net it expects (0.70) and under 0.70 - 0.02 loyalty for most types."""
import numpy as np

NAME = "GT undercut net 0.65"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    return np.where(p <= ctx["tau"], ctx["below"](p), 0.65)
