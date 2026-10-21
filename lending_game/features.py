"""Feature engineering shared by all PD models."""
import numpy as np
import pandas as pd

EMPLOYMENT = ["employed", "student", "unemployed", "retired"]

# Income segments: the amount of information competitors hold that we don't
# is almost entirely a function of income (std of hidden PD: 7pp at 0, 0.3pp above 40k).
INCOME_EDGES = [-1, 0, 5_000, 10_000, 20_000, 40_000, np.inf]
INCOME_LABELS = ["0", "1-5k", "5-10k", "10-20k", "20-40k", "40k+"]


def income_segment(income: pd.Series) -> np.ndarray:
    return pd.cut(income, INCOME_EDGES, labels=INCOME_LABELS).astype(str).to_numpy()


def make_features(df: pd.DataFrame, signals: list[str], use_sex: bool = True) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if use_sex:
        out["female"] = (df["sex"] == "F").astype(int)
    for e in EMPLOYMENT:
        out[f"emp_{e}"] = (df["employment"] == e).astype(int)
    out["married"] = df["married"].astype(int)
    out["income"] = df["income"].astype(float)
    out["log_income"] = np.log1p(df["income"].astype(float))
    for s in signals:
        out[s] = df[s].astype(float)
        # A signal of exactly 0 (21% of rows) behaves like "no signal": default rate is the
        # population average, not the worst. Trees find this alone, the flag makes it explicit.
        out[f"{s}_missing"] = (df[s] == 0).astype(int)
    return out
