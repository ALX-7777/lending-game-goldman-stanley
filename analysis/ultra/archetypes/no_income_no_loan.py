"""Competitor archetype: business rule 'no income, no loan'; else break-even + 3%, no offer above PD 0.25.

A base-like team that adds the credit-policy rule any bank-minded student would write: applicants
with income exactly 0 (segment label "0": 17% of applicants, 35% of our band, default rate 20%,
break-even 0.25) are refused regardless of PD. Everybody else gets break-even + 3% on the
competitor's own PD estimate, with no offer above PD 0.25.

Why it matters: the grid's other reject worlds reject on the competitor's OWN PD, so the pool we
inherit is selected on their signals (cursed). Rejection on a VISIBLE attribute (income) leaves an
uncursed pool for our safety net.
"""
import numpy as np

NAME = "no loan if income = 0; else +3%, reject >25%"
REGIME = "reject"
ALSO_VS_BASE = True

OWN_THRESHOLD = 0.25   # competitor's own PD cut-off (not ctx['tau'])
MARGIN = 0.03          # break-even + 3%
REFUSED_SEGMENT = "0"  # income exactly 0


def rates(p, ctx):
    """Offered rate per applicant given the competitor's own PD estimate p; np.inf = no offer."""
    p = np.asarray(p, dtype=float)
    be = p / (1.0 - p)
    r = np.where(p <= OWN_THRESHOLD, np.clip(be + MARGIN, 0.0, 1.0), np.inf)
    seg = np.asarray(ctx["seg"])
    return np.where(seg == REFUSED_SEGMENT, np.inf, r).astype(float)
