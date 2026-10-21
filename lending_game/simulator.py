"""Market simulator implementing the rules of the game.

Applicant i of type t picks the lowest of r(1,i) - 0.02*[t=1], r(2,i) - 0.02*[t=2], r(3,i) - 0.02*[t=3].
No offer = np.inf. Profit on a loan: +10,000*r if repaid, -10,000 if default.
"""
import numpy as np

LOAN = 10_000.0
PREF = 0.02
US = 1  # we are lender 2 -> column index 1


def draw_types(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, n)


def effective_rates(rates: np.ndarray, types: np.ndarray, rng) -> np.ndarray:
    eff = np.array(rates, dtype=float, copy=True)
    eff[np.arange(len(types)), types] -= PREF
    eff += rng.uniform(0, 1e-9, eff.shape)  # random tie-break; inf stays inf
    return eff


def simulate(rates: np.ndarray, types: np.ndarray, y: np.ndarray, pF: np.ndarray, seed: int = 0):
    """rates: (n, 3) array with np.inf for 'no offer'.
    Returns (per-lender results, choice array with -1 = no loan).
    'realized' uses actual defaults; 'expected' uses the full-information PD (smoother)."""
    rng = np.random.default_rng(seed)
    eff = effective_rates(rates, types, rng)
    best = eff.min(axis=1)
    choice = np.where(np.isfinite(best), eff.argmin(axis=1), -1)
    out = []
    for k in range(3):
        won = choice == k
        r = rates[won, k]
        out.append({
            "share": float(won.mean()),
            "loans": int(won.sum()),
            "profit_realized": float(np.where(y[won] == 1, -LOAN, LOAN * r).sum()),
            "profit_expected": float((LOAN * ((1 - pF[won]) * r - pF[won])).sum()),
            "default_realized": float(y[won].mean()) if won.any() else np.nan,
            "default_expected": float(pF[won].mean()) if won.any() else np.nan,
            "mean_rate": float(r.mean()) if won.any() else np.nan,
        })
    return out, choice


class FastEvaluator:
    """Precomputes, for every (scenario, type draw), the best effective competitor rate faced by
    each applicant, so evaluating one candidate pricing rule for us is one vectorized comparison."""

    def __init__(self, comp_rates: dict, n: int, seeds: list[int], pF: np.ndarray):
        self.pF = pF
        self.cases = []  # (scenario, our_fan mask, min competitor effective rate)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            types = draw_types(n, seed)
            for name, (r1, r3) in comp_rates.items():
                eff1 = r1 - PREF * (types == 0) + rng.uniform(0, 1e-9, n)
                eff3 = r3 - PREF * (types == 2) + rng.uniform(0, 1e-9, n)
                self.cases.append((name, types == US, np.minimum(eff1, eff3)))

    def profits(self, r2: np.ndarray) -> dict:
        """Expected profit (full-information PD) per scenario, one entry per type draw."""
        res = {}
        for name, our_fan, comp_min in self.cases:
            eff2 = np.where(np.isfinite(r2), r2 - PREF * our_fan, np.inf)
            won = eff2 < comp_min
            prof = LOAN * ((1 - self.pF[won]) * r2[won] - self.pF[won]).sum()
            res.setdefault(name, []).append(prof)
        return res

    def objective(self, r2: np.ndarray):
        per_scn = {k: float(np.mean(v)) for k, v in self.profits(r2).items()}
        return float(np.mean(list(per_scn.values()))), float(min(per_scn.values())), per_scn
