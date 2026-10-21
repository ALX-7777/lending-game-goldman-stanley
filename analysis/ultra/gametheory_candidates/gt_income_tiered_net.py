"""Rules-lawyer rate card: the winner's curse is concentrated at low income, so the net is tiered by income
segment instead of by PD: 0.99 at income 0, 0.75 at 1-5k, 0.60 at 5-10k, 0.45 at 10-20k, 0.99 above 20k."""
import numpy as np

NAME = "GT income-tiered net 0.99/0.75/0.60/0.45"
REGIME = "net"
ALSO_VS_BASE = True
TIERS = {"0": 0.99, "1-5k": 0.75, "5-10k": 0.60, "10-20k": 0.45, "20-40k": 0.99, "40k+": 0.99}


def rates(p, ctx):
    net = np.array([TIERS[s] for s in ctx["seg"]])
    return np.where(p <= ctx["tau"], ctx["below"](p), net)
