"""Archetype: the lecture formula taken literally, R = f(X) + M with a 'conservative' M = 10%, everyone served.

Slide 14 of Session 8 writes R = f(X) + M with interest paid at issuance (break-even R = p, not p/(1-p)) and
slide 10 says to bid conservatively, so a literal team prices every applicant at PD + 0.10: no rejection, no net.
Identical to analysis/ultra/lens_student/student_lecture_p_plus_10.py except ALSO_VS_BASE = False (the asymmetric
pairing with 'reject >25%' is inert for the band).
"""
import numpy as np

NAME = "lecture formula p + 10% to everyone"
REGIME = "cheap"
ALSO_VS_BASE = False


def rates(p, ctx):
    """Rate = PD + 0.10 for every applicant (clipped to the 100% cap); never np.inf, deterministic."""
    return np.minimum(np.asarray(p, float) + 0.10, 1.0)
