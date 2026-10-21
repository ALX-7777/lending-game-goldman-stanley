"""Rules-lawyer lens: snipe just under the rival net it expects (0.85) -> flat 0.83 on the band.
This lands inside the +/-0.02 loyalty window around our 0.828, so the band splits by applicant type."""
import numpy as np

NAME = "GT snipe net 0.83 (just under 0.85)"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    return np.where(p <= ctx["tau"], ctx["below"](p), 0.83)
