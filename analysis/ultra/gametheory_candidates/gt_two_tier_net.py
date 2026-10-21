"""Game-theorist lens: two-tier last resort. Moderately risky band members (PD up to 0.30) get a cheap net 0.55
to capture the rent the team expects rivals to leave at 0.90-0.99; the very risky (PD > 0.30) get the cap-ish 0.99
(a rules lawyer never writes 'no offer': 0.99 weakly dominates it)."""
import numpy as np

NAME = "GT two-tier net 0.55 to PD 0.30, 0.99 above"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    band = np.where(p <= 0.30, 0.55, 0.99)
    return np.where(p <= ctx["tau"], ctx["below"](p), band)
