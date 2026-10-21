"""Competitor archetype: the plausible student rate card 'risky people pay a flat 50%'.

Break-even + 3% (p/(1-p) + 0.03, capped at 1.0) up to the competitor's OWN threshold
PD 0.15 (same threshold for every income segment, not our ctx['tau']), then a flat
0.50 for everybody riskier. No rejection, no safety net: every applicant receives an
offer. Deterministic, numpy only.

Relative to our rule this competitor is cheap (be + 3%) on the part of our risky band
with own PD in (tau, 0.15] and charges 0.50 above 0.15 (model + 0.32 at PD 0.15,
+ 0.17 at PD 0.25, - 0.04 at PD 0.35). Inside the flat zone its price no longer moves
with its private signal, so the within-band winner's curse is confined to the 0.15
boundary. Paired with the base 'reject > 25%' competitor the world is inert, hence
ALSO_VS_BASE = False.
"""
import numpy as np

NAME = "flat 0.50 above own PD 0.15"
REGIME = "margin"
ALSO_VS_BASE = False
X, T = 0.50, 0.15


def rates(p, ctx):
    """Offered rate per applicant given the competitor's own PD estimate p (array in (0,1))."""
    p = np.asarray(p, float)
    return np.where(p <= T, np.minimum(p / (1 - p) + 0.03, 1.0), X)
