"""Build a SYNTHETIC round-1 diagnostic in the instructor's type-2 layout, from our simulator on the past loans, to test
analysis/round1_diagnostic.py before our real file arrives.  Four type-1 and four type-3 rival strategies are drawn from a
mixture of profiles; each applicant faces a random pair; our offers are the submitted rule applied to the past loans;
defaults are the realized past outcomes.
Usage, from the project folder:  python analysis/make_fake_diagnostic.py <out_dir>
Writes <out_dir>/fake_diagnostic.csv and <out_dir>/fake_scores.csv (id, segment, p2, rate)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_payoff_matrix as bpm  # noqa: E402
import rivals  # noqa: E402
from lending_game.simulator import LOAN, PREF  # noqa: E402

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)
n, y, p1, p2, p3, seg = bpm.n, bpm.y, bpm.p1, bpm.p2, bpm.p3, bpm.seg
tau_arr, m_arr, be = bpm.tau_arr, bpm.m_arr, bpm.be
floor = pd.read_csv(bpm.OUT / "curse_floor.csv")
adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
safe = be(p2) + m_arr
band = np.minimum(adj + 0.05, 0.989)
w = 1 / (1 + np.exp(-(p2 - tau_arr - 0.01) / 0.01))
own = np.round(safe + w * (band - safe), 4)

rng = np.random.default_rng(123)
profiles = ["likeus", "smart", "netters", "modelfollowers"]
R1 = [rivals.draw_rival(rng, p1, list(rivals.MIXTURES[pr].values())[0], pr, tau_arr, n)[0] for pr in profiles]
R3 = [rivals.draw_rival(rng, p3, list(rivals.MIXTURES[pr].values())[0], pr, tau_arr, n)[0] for pr in profiles]
pick1, pick3 = rng.integers(0, 4, n), rng.integers(0, 4, n)
c1 = np.choose(pick1, R1)
c3 = np.choose(pick3, R3)
types = rng.integers(1, 4, n)
e1 = np.where(np.isfinite(c1), c1 - PREF * (types == 1), np.inf)
e2 = own - PREF * (types == 2)
e3 = np.where(np.isfinite(c3), c3 - PREF * (types == 3), np.inf)
accepted = (e2 < np.minimum(e1, e3)).astype(int)
profit = np.where(accepted == 1, ((1 - y) * own - y) * LOAN, np.nan)
ids = np.arange(1, n + 1)
pd.DataFrame({"id": ids, "own": own, "competing1": np.where(np.isfinite(c1), c1, np.nan), "competing3": np.where(np.isfinite(c3), c3, np.nan),
              "borrowertype": types, "accepted": accepted, "default": y.astype(int), "profit": profit}).to_csv(out_dir / "fake_diagnostic.csv", index=False)
pd.DataFrame({"id": ids, "segment": seg, "p2": p2, "rate": own}).to_csv(out_dir / "fake_scores.csv", index=False)
print(f"written {out_dir / 'fake_diagnostic.csv'} and fake_scores.csv: share {accepted.mean()*100:.1f}%, profit {np.nansum(profit)/1e6:.1f} M, rival profiles {profiles}")
