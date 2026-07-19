from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.schema import parse_pair, validate_quotes
from fxbacktest.data.synthetic import SyntheticFxDataGenerator


def test_parse_pair_splits_base_and_quote():
    assert parse_pair("EURUSD") == ("EUR", "USD")
    assert parse_pair("USDJPY") == ("USD", "JPY")
    assert parse_pair("EURHUF") == ("EUR", "HUF")


def test_parse_pair_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_pair("EUR")
    with pytest.raises(ValueError):
        parse_pair("EURUSDX")


def _multi_pair_quotes():
    df1 = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-03-31", seed=1).generate()
    df2 = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                   seed=2, base_spot=110.0).generate()
    return df1, df2


def test_validate_quotes_accepts_genuinely_multi_pair_data():
    df1, df2 = _multi_pair_quotes()
    combined = pd.concat([df1, df2], ignore_index=True)
    validate_quotes(combined)  # should not raise


def test_validate_quotes_catches_per_pair_tenor_gap():
    """A tenor gap in ONE pair must be caught even though the OTHER pair's
    rows for that same date fill in a complete tenor set — grouping by date
    alone (not (pair, date)) would incorrectly mask this."""
    df1, df2 = _multi_pair_quotes()
    first_date = df1["date"].iloc[0]
    # drop one tenor row for df1 on its first date only
    mask = ~((df1["date"] == first_date) & (df1["tenor"] == "1Y"))
    df1_broken = df1[mask]
    combined = pd.concat([df1_broken, df2], ignore_index=True)

    with pytest.raises(ValueError, match="incomplete tenor set"):
        validate_quotes(combined)


def test_validate_quotes_catches_per_pair_stale_run():
    """A stale (repeated) atm_vol run in one pair/tenor must be caught even
    when interleaved with another pair's non-stale rows for the same tenor —
    grouping by tenor alone (not (pair, tenor)) would corrupt the run-length
    detection across pairs."""
    df1, df2 = _multi_pair_quotes()
    df1 = df1.copy()
    stale_mask = (df1["tenor"] == "1M")
    stale_dates = df1.loc[stale_mask, "date"].sort_values().iloc[:12].tolist()
    df1.loc[stale_mask & df1["date"].isin(stale_dates), "atm_vol"] = 0.08

    combined = pd.concat([df1, df2], ignore_index=True)
    with pytest.raises(ValueError, match="stale atm_vol"):
        validate_quotes(combined)
