"""Segment-dependent safety net: a team that simulates the market computes, per income segment, the
default rate of the pool its rivals refuse (0.39-0.42 at low income when rivals reject above 25%, i.e.
break-even 0.64-0.73; much smaller hidden information above 10k) and sets net = break-even + a buffer,
decreasing with income: 0.75 / 0.70 / 0.70 / 0.55 / 0.45 / 0.45."""
import numpy as np

NAME = "segment nets .75/.70/.70/.55/.45"
REGIME = "net"
ALSO_VS_BASE = True
NETS = {"0": 0.75, "1-5k": 0.70, "5-10k": 0.70, "10-20k": 0.55, "20-40k": 0.45, "40k+": 0.45}


def rates(p, ctx):
    net = np.array([NETS[s] for s in ctx["seg"]])
    return np.where(p <= ctx["tau"], ctx["below"](p), net)
