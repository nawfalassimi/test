from __future__ import annotations

import pandas as pd

QUOTE_COLUMNS = [
    "date", "tenor", "spot", "fwd_points",
    "atm_vol", "rr25", "bf25", "rr10", "bf10",
]

TENORS = ["1W", "1M", "3M", "6M", "1Y"]

TENOR_DAYS = {"1W": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365}
TENOR_YEARS = {tenor: days / 365.0 for tenor, days in TENOR_DAYS.items()}


def validate_quotes(df: pd.DataFrame) -> None:
    """Raise ValueError on structural or economic problems with a quotes DataFrame.

    Checks: required columns present, every date has exactly the full tenor set,
    no NaNs, no repeated (stale) atm_vol runs, and no inverted RR/BF signs
    (bf25 must be non-negative, bf10 must be >= bf25 since 10-delta wings should
    be priced at least as rich as 25-delta wings).
    """
    missing_cols = set(QUOTE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"quotes missing required columns: {sorted(missing_cols)}")

    if df[QUOTE_COLUMNS].isna().any().any():
        raise ValueError("quotes contain NaNs")

    tenor_sets = df.groupby("date")["tenor"].apply(lambda s: frozenset(s))
    bad_dates = tenor_sets[tenor_sets != frozenset(TENORS)]
    if len(bad_dates) > 0:
        raise ValueError(f"dates with incomplete tenor set: {list(bad_dates.index[:5])}")

    if (df["bf25"] < 0).any() or (df["bf10"] < 0).any():
        raise ValueError("negative butterfly quote found (BF must be non-negative)")

    if (df["bf10"] < df["bf25"]).any():
        raise ValueError("bf10 < bf25 found (10-delta wings should cost at least as much as 25-delta)")

    stale_run_threshold = 10
    for tenor, group in df.sort_values("date").groupby("tenor"):
        run_lengths = group["atm_vol"].diff().eq(0).astype(int)
        run_lengths = run_lengths.groupby((run_lengths != run_lengths.shift()).cumsum()).cumsum()
        if (run_lengths >= stale_run_threshold).any():
            raise ValueError(f"stale atm_vol detected for tenor {tenor} ({stale_run_threshold}+ repeated days)")
