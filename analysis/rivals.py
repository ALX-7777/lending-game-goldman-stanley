"""Random rival generators for the Monte Carlo analyses.

Profiles (each rival is drawn independently; its rates are a function of ITS OWN PD estimate p):
  mixed          : broad, naive and sophisticated alike (cheap lending to the band, the lecture-formula mistake, noisy
                   models, rejecters, netters, rampers).
  smart          : curse-aware (calibrated model, correct break-even, low thresholds); band = net / ramp / margin /
                   data-driven schedule / reject / compromise.
  clones         : OUR method with different parameters: thresholds = ours x U(0.6, 2.0) or U(0.06, 0.25), safe margins
                   U(0.02, 0.05), band = a ramp (L ~ U(0.5, 0.8), H ~ U(0.8, 0.99)) / the data-driven schedule shifted
                   U(-0.10, +0.05) / a flat net U(0.65, 0.99).
  modelfollowers : model + margin for everyone, no net: constant margin U(0.02, 0.06), or growing with risk
                   (3% + k (PD - 5%), k ~ U(0.3, 2)), 30% of them refusing above U(0.25, 0.40).
  netters        : low thresholds (ours or U(0.08, 0.20)); net = an obvious number (1.00/0.99/0.95/0.90/0.85/0.828) /
                   a break-even-derived level U(0.65, 0.90) / two-tier (lower net on the low band).
  verysmart      : rivals that best-respond to US: the data-driven schedule fitted against smart rivals, shifted DOWN by
                   U(0.00, 0.10); safe margins U(0.015, 0.035) (about one point below ours); our thresholds; with
                   probability 0.3 the compromise (model + U(0.20, 0.30)) on the low band.
All smart profiles use calibrated models (logit noise sd <= 0.12) and the correct break-even formula.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REGIMES = {"mixed": ["cheap", "margin", "reject", "net", "ramp"],
           "smart": ["net", "ramp", "margin", "datadriven", "reject", "compromise"],
           "clones": ["ramp", "datadriven", "flatnet"],
           "modelfollowers": ["constant", "growing", "constant_reject"],
           "netters": ["obvious", "breakeven", "twotier"],
           "verysmart": ["undercut_dd", "undercut_dd_compromise", "undercut_ramp"],
           "likeus": ["floor_margin", "floor_margin_reject", "floor_margin_cap"]}
MIXTURES = {
    "mixed": {"baseline (45/15/15/15/10)": [0.45, 0.15, 0.15, 0.15, 0.10],
              "naive rivals (65/10/15/7/3)": [0.65, 0.10, 0.15, 0.07, 0.03],
              "sophisticated rivals (25/15/10/30/20)": [0.25, 0.15, 0.10, 0.30, 0.20]},
    "smart": {"smart (net 25 / ramp 20 / margin 15 / data-driven 15 / reject 15 / compromise 10)": [0.25, 0.20, 0.15, 0.15, 0.15, 0.10],
              "smart, net-heavy (40 / 25 / 10 / 10 / 10 / 5)": [0.40, 0.25, 0.10, 0.10, 0.10, 0.05],
              "smart, margin-heavy (15 / 15 / 35 / 15 / 10 / 10)": [0.15, 0.15, 0.35, 0.15, 0.10, 0.10]},
    "clones": {"clones of our method (ramp 45 / data-driven 30 / flat net 25)": [0.45, 0.30, 0.25]},
    "modelfollowers": {"model-followers (constant 40 / growing 40 / constant+reject 20)": [0.40, 0.40, 0.20]},
    "netters": {"netters (obvious 40 / break-even-derived 40 / two-tier 20)": [0.40, 0.40, 0.20]},
    "verysmart": {"very smart: best-responders (undercut data-driven 50 / + compromise 30 / undercut ramp 20)": [0.50, 0.30, 0.20]},
    "likeus": {"like us, no net: curse-adjusted break-even + margin (everyone 50 / refuse >30% PD 30 / cap at 0.60 20)": [0.50, 0.30, 0.20]},
}


def _curse_floor():
    """The structural curse floor fitted by analysis/curse_adjusted_breakeven.py: (PD grid, adjusted break-even)."""
    f = ROOT / "output" / "round1" / "curse_floor.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    return t["pd"].to_numpy(), t["adjusted_breakeven"].to_numpy()


FLOOR = _curse_floor()


def _dd_schedule(suffix=""):
    """A data-driven per-category schedule from a previous fit, as (edges, rates); None if not fitted yet."""
    f = ROOT / "output" / "round1" / f"breakeven_ramp{suffix}.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    lo = [float(s.split("-")[0].rstrip("%")) / 100 for s in t["PD category"]]
    hi = [float(s.split("-")[1].rstrip("%")) / 100 for s in t["PD category"]]
    return np.array(lo + [hi[-1]]), t["data-driven rate"].to_numpy()


DD = _dd_schedule("")            # fitted against the 'mixed' profile
DD_SMART = _dd_schedule("_smart") or DD  # fitted against the 'smart' profile (what a very smart rival would compute)


def _dd_rates(q, dd, shift):
    edges, rates = dd
    idx = np.clip(np.digitize(q, edges[1:-1]), 0, len(rates) - 1)
    return np.clip(rates[idx] + shift, 0.10, 1.0)


def draw_rival(rng, p, W, profile, tau_arr, n):
    """Returns (rates array, regime label, short description)."""
    be = lambda x: x / (1 - x)  # noqa: E731
    q = np.clip(p, 1e-4, 1 - 1e-4)
    regimes = REGIMES[profile]
    regime = regimes[rng.choice(len(regimes), p=W)]

    if profile == "mixed":
        sd = rng.uniform(0, 0.3)
        q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + sd * rng.standard_normal(n))))
        m_safe = rng.uniform(0.01, 0.05)
        slide = rng.random() < 0.10
        safe = np.minimum((q + m_safe) if slide else (be(q) + m_safe), 1.0)
        if regime == "cheap":
            t = rng.choice([np.inf, 0.20, 0.25, 0.30, 0.35])
            return np.where(q <= t, safe, np.inf), regime, f"cheap t={t}"
        if regime == "margin":
            M = rng.uniform(0.08, 0.40)
            return np.where(q <= tau_arr, safe, np.minimum(be(q) + M, 1.0)), regime, f"margin M={M:.2f}"
        if regime == "reject":
            if rng.random() < 0.7:
                t = rng.uniform(0.10, 0.30)
                return np.where(q <= t, safe, np.inf), regime, f"reject t={t:.2f}"
            c = rng.uniform(0.25, 0.50)
            return np.where(safe <= c, safe, np.inf), regime, f"cap c={c:.2f}"
        if regime == "net":
            thr = rng.choice(["ours", 0.15, 0.20, 0.25])
            tt = tau_arr if thr == "ours" else float(thr)
            X = rng.choice([1.0, 0.99, 0.95, 0.90, 0.85, 0.80]) if rng.random() < 0.5 else rng.uniform(0.60, 0.99)
            return np.where(q <= tt, safe, X), regime, f"net X={X:.3f} thr={thr}"
        thr = rng.choice(["ours", 0.15, 0.20])
        tt = tau_arr if thr == "ours" else float(thr)
        L = rng.uniform(0.50, 0.80)
        H = rng.uniform(max(L + 0.05, 0.80), 0.99)
        frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
        return np.where(q <= tt, safe, L + (H - L) * frac), regime, f"ramp {L:.2f}->{H:.2f} thr={thr}"

    # ---------------- all smart profiles: calibrated model, correct break-even
    sd = rng.uniform(0, 0.12)
    q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + sd * rng.standard_normal(n))))

    if profile == "smart":
        m_safe = rng.uniform(0.02, 0.045)
        safe = np.minimum(be(q) + m_safe, 1.0)
        tt = tau_arr if rng.random() < 0.5 else np.full(n, rng.uniform(0.08, 0.20))
        frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
        if regime == "net":
            X = rng.choice([1.0, 0.99, 0.95, 0.90, 0.85, 0.828]) if rng.random() < 0.4 else rng.uniform(0.70, 0.90)
            return np.where(q <= tt, safe, X), regime, f"net X={X:.3f}"
        if regime == "ramp":
            L = rng.uniform(0.55, 0.80)
            H = rng.uniform(max(L + 0.05, 0.80), 0.99)
            return np.where(q <= tt, safe, L + (H - L) * frac), regime, f"ramp {L:.2f}->{H:.2f}"
        if regime == "margin":
            if rng.random() < 0.5:
                M = rng.uniform(0.10, 0.35)
                return np.where(q <= tt, safe, np.minimum(be(q) + M, 1.0)), regime, f"margin M={M:.2f}"
            k = rng.uniform(0.5, 2.0)
            return np.where(q <= tt, safe, np.minimum(be(q) + 0.03 + k * np.maximum(q - 0.05, 0), 1.0)), regime, f"margin k={k:.2f}"
        if regime == "datadriven":
            if DD is None:
                return np.where(q <= tt, safe, 0.60 + 0.30 * frac), regime, "datadriven (fallback ramp)"
            shift = rng.uniform(-0.08, 0.05)
            return np.where(q <= tt, safe, _dd_rates(q, DD, shift)), regime, f"datadriven shift={shift:+.2f}"
        if regime == "reject":
            return np.where(q <= tt, safe, np.inf), regime, "reject band"
        M, cut, X = rng.uniform(0.15, 0.30), rng.uniform(0.13, 0.20), rng.uniform(0.70, 0.95)
        return np.where(q <= tt, safe, np.where(q <= cut, np.minimum(be(q) + M, 1.0), X)), regime, f"compromise M={M:.2f} cut={cut:.2f} net={X:.2f}"

    if profile == "clones":
        m_safe = rng.uniform(0.02, 0.05)
        safe = np.minimum(be(q) + m_safe, 1.0)
        tt = tau_arr * rng.uniform(0.6, 2.0) if rng.random() < 0.6 else np.full(n, rng.uniform(0.06, 0.25))
        frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
        if regime == "ramp":
            L = rng.uniform(0.50, 0.80)
            H = rng.uniform(max(L + 0.05, 0.80), 0.99)
            return np.where(q <= tt, safe, L + (H - L) * frac), regime, f"clone ramp {L:.2f}->{H:.2f} m={m_safe:.3f}"
        if regime == "datadriven":
            dd = DD_SMART if DD_SMART is not None else DD
            if dd is None:
                return np.where(q <= tt, safe, 0.60 + 0.30 * frac), regime, "clone datadriven (fallback)"
            shift = rng.uniform(-0.10, 0.05)
            return np.where(q <= tt, safe, _dd_rates(q, dd, shift)), regime, f"clone datadriven shift={shift:+.2f}"
        X = rng.uniform(0.65, 0.99)
        return np.where(q <= tt, safe, X), regime, f"clone flat net {X:.2f}"

    if profile == "modelfollowers":
        if regime == "constant":
            m = rng.uniform(0.02, 0.06)
            return np.minimum(be(q) + m, 1.0), regime, f"model + {m:.3f} everyone"
        if regime == "growing":
            k = rng.uniform(0.3, 2.0)
            return np.minimum(be(q) + 0.03 + k * np.maximum(q - 0.05, 0), 1.0), regime, f"model + 3% + {k:.2f}(PD-5%)"
        m, t = rng.uniform(0.02, 0.06), rng.uniform(0.25, 0.40)
        return np.where(q <= t, np.minimum(be(q) + m, 1.0), np.inf), regime, f"model + {m:.3f}, reject > {t:.2f}"

    if profile == "netters":
        m_safe = rng.uniform(0.02, 0.045)
        safe = np.minimum(be(q) + m_safe, 1.0)
        tt = tau_arr if rng.random() < 0.5 else np.full(n, rng.uniform(0.08, 0.20))
        if regime == "obvious":
            X = rng.choice([1.0, 0.99, 0.95, 0.90, 0.85, 0.828])
            return np.where(q <= tt, safe, X), regime, f"net {X:.3f} (obvious)"
        if regime == "breakeven":
            X = rng.uniform(0.65, 0.90)
            return np.where(q <= tt, safe, X), regime, f"net {X:.3f} (break-even derived)"
        X_low, X_high, cut = rng.uniform(0.50, 0.75), rng.uniform(0.80, 0.99), rng.uniform(0.15, 0.22)
        return np.where(q <= tt, safe, np.where(q <= cut, X_low, X_high)), regime, f"two-tier net {X_low:.2f}/{X_high:.2f} cut={cut:.2f}"

    if profile == "likeus":
        # curse-aware rivals who price the band at THEIR curse-adjusted break-even + a margin, and never net
        m_safe = rng.uniform(0.02, 0.045)
        safe = np.minimum(be(q) + m_safe, 1.0)
        tt = tau_arr if rng.random() < 0.5 else np.full(n, rng.uniform(0.08, 0.20))
        M = rng.uniform(0.03, 0.25)
        if FLOOR is None:
            adj = be(q) + 0.10  # fallback: a crude curse premium
        else:
            adj = np.interp(q, FLOOR[0], FLOOR[1])
        band = np.minimum(adj + M, 1.0)
        if regime == "floor_margin":
            return np.where(q <= tt, safe, band), regime, f"like us: floor + {M:.2f}"
        if regime == "floor_margin_reject":
            return np.where(q <= tt, safe, np.where(q <= 0.30, band, np.inf)), regime, f"like us: floor + {M:.2f}, refuse >30%"
        return np.where(q <= tt, safe, np.where(band <= 0.60, band, np.inf)), regime, f"like us: floor + {M:.2f}, cap 0.60"

    # verysmart: best-responders to us
    m_safe = rng.uniform(0.015, 0.035)
    safe = np.minimum(be(q) + m_safe, 1.0)
    tt = tau_arr
    frac = np.clip((q - tt) / np.maximum(0.25 - tt, 1e-9), 0, 1)
    dd = DD_SMART if DD_SMART is not None else DD
    shift = -rng.uniform(0.0, 0.10)
    if regime == "undercut_dd" and dd is not None:
        return np.where(q <= tt, safe, _dd_rates(q, dd, shift)), regime, f"undercut data-driven by {-shift:.2f}, m={m_safe:.3f}"
    if regime == "undercut_dd_compromise" and dd is not None:
        M, cut = rng.uniform(0.20, 0.30), rng.uniform(0.13, 0.18)
        band = np.where(q <= cut, np.minimum(be(q) + M, 1.0), _dd_rates(q, dd, shift))
        return np.where(q <= tt, safe, band), regime, f"compromise M={M:.2f} + undercut data-driven by {-shift:.2f}"
    L, H = 0.65 + shift, 0.828 + shift
    return np.where(q <= tt, safe, L + (H - L) * frac), regime, f"undercut our ramp by {-shift:.2f}, m={m_safe:.3f}"
