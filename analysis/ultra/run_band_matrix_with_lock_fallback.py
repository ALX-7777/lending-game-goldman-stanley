"""Run analysis/band_payoff_matrix.py verbatim (as __main__), with one safety net: if an output CSV in
output/round1 is locked by another process (e.g. open in Excel), that single file is written to
analysis/ultra/<same name> instead and a note goes to stderr.  The script itself is not modified.
Usage (from the project folder):  python analysis/ultra/run_band_matrix_with_lock_fallback.py
"""
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "band_payoff_matrix.py"
FALLBACK = ROOT / "analysis" / "ultra"

_orig_to_csv = pd.DataFrame.to_csv


def _to_csv_with_fallback(self, path_or_buf=None, *args, **kwargs):
    try:
        return _orig_to_csv(self, path_or_buf, *args, **kwargs)
    except PermissionError as e:
        alt = FALLBACK / Path(path_or_buf).name
        print(f"[lock-fallback] {path_or_buf} is locked ({e.strerror}); written to {alt} instead",
              file=sys.stderr, flush=True)
        return _orig_to_csv(self, alt, *args, **kwargs)


pd.DataFrame.to_csv = _to_csv_with_fallback
sys.argv = [str(SCRIPT)]
runpy.run_path(str(SCRIPT), run_name="__main__")
