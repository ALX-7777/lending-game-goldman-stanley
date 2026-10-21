"""Our pricing rule and its optimizer.

Rule for applicant i with our PD estimate p and income segment s:
    r_i = p/(1-p) + m(group_i)          if p <= tau(s)     (break-even + a group-specific margin)
    r_i = tail_rate  (or no offer)      if p >  tau(s)     ('lender of last resort' safety net)
Parameters are chosen by coordinate descent to maximize expected profit averaged over
competitor scenarios, each simulated on the past loans where all three signals are known.
"""
import copy

import numpy as np

MARGIN_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.10]
TAU_GRID = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]
TAIL_GRID = [None, 0.50, 0.60, 0.70, 0.80, 0.90, 0.989]


class Rule:
    def __init__(self, p2, seg, keys):
        self.p2, self.seg, self.keys = p2, seg, keys
        self.key_masks = {k: keys == k for k in sorted(set(keys))}
        self.seg_masks = {s: seg == s for s in sorted(set(seg))}

    def rates(self, params: dict) -> np.ndarray:
        m = np.zeros_like(self.p2)
        for k, mask in self.key_masks.items():
            m[mask] = params["margins"][k]
        tau = np.zeros_like(self.p2)
        for s, mask in self.seg_masks.items():
            tau[mask] = params["tau"][s]
        r = np.minimum(self.p2 / (1 - self.p2) + m, 1.0)
        tail = params["tail_rate"]
        if tail is None:
            band = np.inf
        elif params.get("ramp_from") is None:
            band = tail
        else:  # lender-of-last-resort rate rising with the PD: ramp_from at the threshold, tail from PD ramp_top on
            top = params.get("ramp_top", 0.25)
            frac = np.clip((self.p2 - tau) / np.maximum(top - tau, 1e-9), 0, 1)
            band = params["ramp_from"] + (tail - params["ramp_from"]) * frac
        return np.where(self.p2 > tau, band, r)


def default_params(rule: Rule, margin=0.03, tau=0.25, tail_rate=0.989, ramp_from=None, ramp_top=0.25) -> dict:
    return {"margins": {k: margin for k in rule.key_masks},
            "tau": {s: tau for s in rule.seg_masks},
            "tail_rate": tail_rate, "ramp_from": ramp_from, "ramp_top": ramp_top}


def optimize(rule: Rule, evaluator, init: dict, max_sweeps=8, min_gain=1_000.0, verbose=True,
             fixed_tail="search"):
    """Coordinate descent on the grids above; objective = mean expected profit across scenarios.
    fixed_tail: "search" to pick the safety-net rate from TAIL_GRID, otherwise a rate (or None for
    'no offer') that is held fixed, e.g. chosen from analysis/net_rate_scenarios.py."""
    params = copy.deepcopy(init)
    if fixed_tail != "search":
        params["tail_rate"] = fixed_tail

    def score(pr):
        return evaluator.objective(rule.rates(pr))[0]

    best = score(params)
    if verbose:
        print(f"  start: {best / 1e6:.3f} M EUR")
    for sweep in range(max_sweeps):
        improved = False
        for k in rule.key_masks:
            for m in MARGIN_GRID:
                trial = copy.deepcopy(params)
                trial["margins"][k] = m
                s = score(trial)
                if s > best + min_gain:
                    best, params, improved = s, trial, True
        for s_ in rule.seg_masks:
            for t in TAU_GRID:
                trial = copy.deepcopy(params)
                trial["tau"][s_] = t
                s = score(trial)
                if s > best + min_gain:
                    best, params, improved = s, trial, True
        for tail in (TAIL_GRID if fixed_tail == "search" else []):
            trial = copy.deepcopy(params)
            trial["tail_rate"] = tail
            s = score(trial)
            if s > best + min_gain:
                best, params, improved = s, trial, True
        if verbose:
            print(f"  sweep {sweep + 1}: {best / 1e6:.3f} M EUR")
        if not improved:
            break
    return params, best
