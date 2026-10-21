"""Game-theorist lens: Bertrand undercut of the rivals' safety nets.
The team anticipates that rivals put a 0.90-0.99 'last resort' rate on the risky band and that whoever is
cheapest there inherits the whole band (PD ~16-20%, pooled break-even ~0.22). It prices the band at a flat
0.49 ('just under 50%'): still a 25-point cushion over the pooled break-even, but below any round net."""
import numpy as np

NAME = "GT undercut net 0.49"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    return np.where(p <= ctx["tau"], ctx["below"](p), 0.49)
