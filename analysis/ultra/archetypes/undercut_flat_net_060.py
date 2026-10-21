"""Undercut flat net (central case of the 'undercut net' family).

A team that serves everybody and puts ONE round last-resort price on its risky band, chosen BELOW the
0.70-0.99 round nets it expects from rivals.  Break-even + 3% up to its OWN PD 0.12 (the same round
threshold for every income segment), then a flat 0.60 on the whole band above it.

Family members (not duplicated here): timid clone net 0.60 above 0.12 (student lens), undercut 0.65 / 0.49
(game theorist), clone net 0.50 (sophisticated), two-tier nets 0.45-0.55 / 0.95-0.99.  Novelty vs the grid:
the grid's lowest flat band price is clone net 0.70; nothing in it tests the 0.40-0.69 range, where every
flat net >= 0.70 and every ramp with L >= 0.60 win nothing in the band and the best response flips to
model + 0.20/0.30 or a ramp starting at L <= 0.40.

NOT a clone: it ignores ctx['tau'] / ctx['below'] on purpose (own threshold 0.12, own +3% margin).
Band members with own PD > 0.12 face a flat 0.60 = model + 0.46 at PD 0.12, + 0.27 at PD 0.25,
+ 0.06 at PD 0.35; the price does not move with the competitor's signal, so an undercut by us is
cursed only at the 0.12 boundary.
"""
import numpy as np

NAME = "flat 0.60 on the whole band (own PD > 0.12)"
REGIME = "net"
ALSO_VS_BASE = True
X, T = 0.60, 0.12


def rates(p, ctx):
    """Offered rate per applicant given the competitor's OWN PD estimate p (array in (0, 1)).

    No np.inf anywhere (everybody is served), no RNG; output shape (len(p),), finite values in [0.03, 1].
    """
    p = np.asarray(p, float)
    safe = np.minimum(p / (1 - p) + 0.03, 1.0)   # break-even + 3% on the part it considers safe
    return np.where(p <= T, safe, X)              # one round last-resort price on the whole band
