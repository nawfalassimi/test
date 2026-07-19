from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.market.market import Market
from fxbacktest.market.snapshot import build_market_snapshot


def _snapshot_for(pair: str, spot: float, date: pd.Timestamp):
    """A minimal single-row-per-tenor quotes_df for one pair/date, just enough
    to build a real MarketSnapshot (we only need its .spot for these tests)."""
    from fxbacktest.data.schema import TENORS

    rows = [
        {"pair": pair, "date": date, "tenor": t, "spot": spot, "fwd_points": 0.0,
         "atm_vol": 0.08, "rr25": 0.0, "bf25": 0.003, "rr10": 0.0, "bf10": 0.005}
        for t in TENORS
    ]
    df = pd.DataFrame(rows)
    return build_market_snapshot(date, df, pair)


DATE = pd.Timestamp("2022-01-03")


def test_usd_rate_is_one_for_usd_itself():
    market = Market(date=DATE, snapshots={})
    assert market.usd_rate("USD") == 1.0


def test_usd_rate_direct_eurusd():
    snap = _snapshot_for("EURUSD", 1.10, DATE)
    market = Market(date=DATE, snapshots={"EURUSD": snap})
    assert market.usd_rate("EUR") == pytest.approx(1.10)
    assert market.usd_rate("USD") == 1.0


def test_usd_rate_inverse_usdjpy():
    snap = _snapshot_for("USDJPY", 110.0, DATE)
    market = Market(date=DATE, snapshots={"USDJPY": snap})
    assert market.usd_rate("JPY") == pytest.approx(1.0 / 110.0)
    assert market.usd_rate("USD") == 1.0


def test_usd_rate_one_hop_cross_eurhuf_via_eurusd():
    eurusd = _snapshot_for("EURUSD", 1.10, DATE)
    eurhuf = _snapshot_for("EURHUF", 400.0, DATE)
    market = Market(date=DATE, snapshots={"EURUSD": eurusd, "EURHUF": eurhuf})

    # USD per HUF = (USD per EUR) / (HUF per EUR) = 1.10 / 400.0
    assert market.usd_rate("HUF") == pytest.approx(1.10 / 400.0)
    # EUR is still directly resolvable
    assert market.usd_rate("EUR") == pytest.approx(1.10)


def test_usd_rate_one_hop_cross_base_leg():
    # currency is the BASE of the loaded pair, and the QUOTE resolves via an
    # inverted direct pair (USDJPY), exercising the "base == currency" branch.
    eurjpy = _snapshot_for("EURJPY", 150.0, DATE)
    usdjpy = _snapshot_for("USDJPY", 110.0, DATE)
    market = Market(date=DATE, snapshots={"EURJPY": eurjpy, "USDJPY": usdjpy})

    # USD per EUR = (USD per JPY) * (JPY per EUR) = (1/110) * 150
    assert market.usd_rate("EUR") == pytest.approx((1.0 / 110.0) * 150.0)


def test_usd_rate_unresolvable_currency_raises():
    snap = _snapshot_for("EURGBP", 0.85, DATE)
    market = Market(date=DATE, snapshots={"EURGBP": snap})
    with pytest.raises(ValueError, match="cannot resolve"):
        market.usd_rate("GBP")
