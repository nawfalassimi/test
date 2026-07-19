from __future__ import annotations

from typing import Tuple

import pandas as pd

QUOTE_COLUMNS = [
    "date", "tenor", "spot", "fwd_points",
    "atm_vol", "rr25", "bf25", "rr10", "bf10",
]

TENORS = ["1W", "1M", "3M", "6M", "1Y"]

TENOR_DAYS = {"1W": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365}
TENOR_YEARS = {tenor: days / 365.0 for tenor, days in TENOR_DAYS.items()}


def parse_pair(pair: str) -> Tuple[str, str]:
    """Split a 6-character currency pair code into (base, quote), e.g.
    "EURUSD" -> ("EUR", "USD"). Assumes standard 3-letter ISO currency codes."""
    if len(pair) != 6:
        raise ValueError(f"expected a 6-character currency pair code, got {pair!r}")
    return pair[:3], pair[3:]


def validate_quotes(df: pd.DataFrame) -> None:
    """Raise ValueError on structural or economic problems with a quotes DataFrame.

    Checks: required columns present, every (pair, date) has exactly the full
    tenor set, no NaNs, no repeated (stale) atm_vol runs per (pair, tenor), and
    no inverted RR/BF signs (bf25 must be non-negative, bf10 must be >= bf25
    since 10-delta wings should be priced at least as rich as 25-delta wings).

    All per-date/per-tenor checks are grouped by "pair" too, since a quotes
    DataFrame may contain more than one currency pair — grouping by date or
    tenor alone would interleave different pairs' data and silently corrupt
    these checks (e.g. one pair's stale-vol run masked by another pair's rows).
    """
    missing_cols = (set(QUOTE_COLUMNS) | {"pair"}) - set(df.columns)
    if missing_cols:
        raise ValueError(f"quotes missing required columns: {sorted(missing_cols)}")

    if df[QUOTE_COLUMNS].isna().any().any():
        raise ValueError("quotes contain NaNs")

    tenor_sets = df.groupby(["pair", "date"])["tenor"].apply(lambda s: frozenset(s))
    bad_dates = tenor_sets[tenor_sets != frozenset(TENORS)]
    if len(bad_dates) > 0:
        raise ValueError(f"(pair, date) combinations with an incomplete tenor set: {list(bad_dates.index[:5])}")

    if (df["bf25"] < 0).any() or (df["bf10"] < 0).any():
        raise ValueError("negative butterfly quote found (BF must be non-negative)")

    if (df["bf10"] < df["bf25"]).any():
        raise ValueError("bf10 < bf25 found (10-delta wings should cost at least as much as 25-delta)")

    stale_run_threshold = 10
    for (pair, tenor), group in df.sort_values("date").groupby(["pair", "tenor"]):
        run_lengths = group["atm_vol"].diff().eq(0).astype(int)
        run_lengths = run_lengths.groupby((run_lengths != run_lengths.shift()).cumsum()).cumsum()
        if (run_lengths >= stale_run_threshold).any():
            raise ValueError(f"stale atm_vol detected for pair {pair}, tenor {tenor} ({stale_run_threshold}+ repeated days)")
