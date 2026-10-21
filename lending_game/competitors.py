"""Competitor archetypes: what might a team with a decent PD model and some margin rule do?

Each archetype maps a competitor's own PD estimate to an offered rate.
  rule="breakeven": r = p/(1-p) + m   (correct break-even in this game: interest only if repaid)
  rule="slide":     r = p + m         (the slide formula, NPV = R - PD; underprices risky borrowers)
reject_above: no offer if p > threshold.  net: rate offered instead of 'no offer' (e.g. 0.99).
"""
import numpy as np

# A selection-aware team would widen margins where hidden information is large (low income).
SMART_MARGINS = {"0": 0.06, "1-5k": 0.05, "5-10k": 0.045, "10-20k": 0.035, "20-40k": 0.025, "40k+": 0.02}

ARCHETYPES = {
    "base":        dict(rule="breakeven", margin=0.03, reject_above=0.25),
    "aggressive":  dict(rule="breakeven", margin=0.01, reject_above=None),
    "cautious":    dict(rule="breakeven", margin=0.05, reject_above=0.20),
    "no_reject":   dict(rule="breakeven", margin=0.03, reject_above=None),
    "slide":       dict(rule="slide", margin=0.03, reject_above=0.25),
    "slide_norej": dict(rule="slide", margin=0.03, reject_above=None),
    "smart":       dict(rule="breakeven", seg_margins=SMART_MARGINS, reject_above=0.25, net=0.99),
    "base_net":    dict(rule="breakeven", margin=0.03, reject_above=0.25, net=0.99),
    "loose":       dict(rule="breakeven", margin=0.02, reject_above=0.30),
    "tight":       dict(rule="breakeven", margin=0.04, reject_above=0.20),
}

# (scenario name, archetype of lender 1, archetype of lender 3)
SCENARIOS = [
    ("base / base", "base", "base"),
    ("aggressive / aggressive", "aggressive", "aggressive"),
    ("cautious / cautious", "cautious", "cautious"),
    ("no_reject / no_reject", "no_reject", "no_reject"),
    ("slide / slide", "slide", "slide"),
    ("slide_norej / slide_norej", "slide_norej", "slide_norej"),
    ("smart / smart", "smart", "smart"),
    ("base_net / base_net", "base_net", "base_net"),
    ("base / aggressive", "base", "aggressive"),
    ("cautious / slide", "cautious", "slide"),
    ("smart / base", "smart", "base"),
    ("loose / tight", "loose", "tight"),
]


def price(p, rule="breakeven", margin=0.03, reject_above=None, net=None, seg=None, seg_margins=None):
    base = p / (1 - p) if rule == "breakeven" else p
    if seg_margins is not None:
        m = np.zeros_like(p)
        for s, v in seg_margins.items():
            m[seg == s] = v
    else:
        m = margin
    r = np.minimum(base + m, 1.0)
    if reject_above is not None:
        r = np.where(p > reject_above, np.inf if net is None else net, r)
    return r


def competitor_rates(p1, p3, seg) -> dict:
    return {name: (price(p1, seg=seg, **ARCHETYPES[a1]), price(p3, seg=seg, **ARCHETYPES[a3]))
            for name, a1, a3 in SCENARIOS}


def clone_rates(p, seg, rule_params: dict, net, tau_scale: float = 1.0):
    """A competitor that reasons like us: applies OUR rule (margins and thresholds by income segment)
    to its own PD estimate, with a safety net at `net`. tau_scale > 1 = a more moderate clone that
    keeps lending at normal rates further into the risky band."""
    from .strategy import Rule  # local import to avoid a circular import

    prm = {"margins": rule_params["margins"],
           "tau": {s: min(1.0, t * tau_scale) for s, t in rule_params["tau"].items()},
           "tail_rate": net}
    return Rule(p, seg, seg).rates(prm)
