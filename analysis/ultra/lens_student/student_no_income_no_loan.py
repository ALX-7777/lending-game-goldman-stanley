"""Student lens: business rule 'no income, no loan' (17% of applicants, 35% of the band, default 20%),
otherwise break-even + 3%, no offer above PD 0.25."""
import numpy as np
NAME = "no loan if income = 0; else +3%, reject >25%"
REGIME = "reject"
ALSO_VS_BASE = True

def rates(p, ctx):
    p = np.asarray(p, float)
    r = np.where(p <= 0.25, np.minimum(p / (1 - p) + 0.03, 1.0), np.inf)
    return np.where(np.asarray(ctx["seg"]) == "0", np.inf, r)
