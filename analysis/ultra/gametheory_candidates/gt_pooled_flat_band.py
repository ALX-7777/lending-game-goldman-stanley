"""Market-share / rank-maximizer lens: the team expects rivals to follow the lecture (reject above 25%) and wants
the whole risky band for itself. It prices the band at a FLAT segment price = pooled break-even of the band in
that income segment + 0.15 (band PD by segment: 0.20 / 0.165 / 0.15 / 0.10 -> 0.40 / 0.35 / 0.32 / 0.26).
A flat price does not depend on the team's own signal, so winning against it carries no within-band curse."""
import numpy as np

NAME = "GT pooled flat band price by income (+0.15)"
REGIME = "margin"
ALSO_VS_BASE = True
FLAT = {"0": 0.40, "1-5k": 0.35, "5-10k": 0.32, "10-20k": 0.26, "20-40k": 0.48, "40k+": 0.48}


def rates(p, ctx):
    flat = np.array([FLAT[s] for s in ctx["seg"]])
    return np.where(p <= ctx["tau"], ctx["below"](p), flat)
