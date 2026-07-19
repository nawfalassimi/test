from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.market.market import Market
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position
from fxbacktest.strategies.base import get_strategy
from fxbacktest.strategies.short_vol_signal import ShortVolSignalStrategy
from fxbacktest.strategies.signals import SignalBundle


@pytest.fixture(scope="module")
def quotes_df():
    eurusd = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-03-31", seed=7).generate()
    usdjpy = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                      seed=8, base_spot=110.0).generate()
    return pd.concat([eurusd, usdjpy], ignore_index=True)


@pytest.fixture(scope="module")
def vix_df(quotes_df):
    dates = quotes_df["date"].drop_duplicates()
    return pd.DataFrame({"date": dates, "vix": 15.0})


def _market_for(date, quotes_df, pair="EURUSD"):
    return Market(date=date, snapshots={pair: build_market_snapshot(date, quotes_df, pair)})


def _fake_bundle(dates, entry_dates=(), exit_dates=()):
    idx = pd.DatetimeIndex(dates)
    return SignalBundle(
        composite_flags=pd.DataFrame(index=idx),
        composite_count=pd.Series(0, index=idx),
        rr_condition=pd.Series(False, index=idx),
        hard_stop_inverted=pd.Series(False, index=idx),
        hard_stop_vix=pd.Series(False, index=idx),
        entry_ok=pd.Series(idx.isin(entry_dates), index=idx),
        exit_now=pd.Series(idx.isin(exit_dates), index=idx),
    )


def test_registered_under_expected_name():
    assert get_strategy("short_vol_signal") is ShortVolSignalStrategy


def test_pairs_kwarg_and_defaults(quotes_df, vix_df):
    s = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df, pairs=["EURUSD", "USDJPY"])
    assert s.required_pairs == ["EURUSD", "USDJPY"]

    s2 = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    assert s2.required_pairs == ["EURUSD"]
    assert s2.tenor == "3M"
    assert s2.target_delta == 0.10


def test_duplicate_pairs_raises(quotes_df, vix_df):
    with pytest.raises(ValueError):
        ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df, pairs=["EURUSD", "EURUSD"])


def test_both_pair_and_pairs_raises(quotes_df, vix_df):
    with pytest.raises(ValueError):
        ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df, pair="USDJPY", pairs=["EURUSD"])


def test_unknown_tenor_raises(quotes_df, vix_df):
    with pytest.raises(ValueError):
        ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df, tenor="2M")


def test_no_entry_when_entry_ok_false(quotes_df, vix_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]
    market = _market_for(date, quotes_df)
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    strategy._signals["EURUSD"] = _fake_bundle(dates)
    portfolio = Portfolio()

    assert strategy.generate_orders(date, market, portfolio) == []


def test_entry_produces_strangle_around_spot(quotes_df, vix_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]
    market = _market_for(date, quotes_df)
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df, target_delta=0.10, tenor="3M")
    strategy._signals["EURUSD"] = _fake_bundle(dates, entry_dates=(date,))
    portfolio = Portfolio()

    orders = strategy.generate_orders(date, market, portfolio)
    assert len(orders) == 2
    assert {o.instrument.option_type for o in orders} == {"call", "put"}
    assert all(o.side == "sell" for o in orders)
    assert all(o.strategy_id == "short_vol_signal" for o in orders)
    clip_ids = {o.clip_id for o in orders}
    assert len(clip_ids) == 1
    assert next(iter(clip_ids)) == f"short_vol_signal_EURUSD_{date:%Y%m%d}"

    call = next(o for o in orders if o.instrument.option_type == "call")
    put = next(o for o in orders if o.instrument.option_type == "put")
    spot = market.snapshot("EURUSD").spot
    # a 10-delta strangle should straddle spot: call strike above, put strike below
    assert call.instrument.strike > spot
    assert put.instrument.strike < spot


def test_no_entry_while_position_already_open_even_if_entry_ok(quotes_df, vix_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]
    market = _market_for(date, quotes_df)
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    strategy._signals["EURUSD"] = _fake_bundle(dates, entry_dates=(date,))
    portfolio = Portfolio()
    portfolio.positions.append(Position(
        instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="prev",
        strategy_id="short_vol_signal", entry_date=date, entry_price=0.0,
    ))

    assert strategy.generate_orders(date, market, portfolio) == []


def test_exit_now_flags_open_position_for_early_close(quotes_df, vix_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]
    market = _market_for(date, quotes_df)
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    strategy._signals["EURUSD"] = _fake_bundle(dates, exit_dates=(date,))
    portfolio = Portfolio()
    portfolio.positions.append(Position(
        instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="prev",
        strategy_id="short_vol_signal", entry_date=date, entry_price=0.0,
    ))

    orders = strategy.generate_orders(date, market, portfolio)
    assert orders == []
    assert portfolio.positions[0].pending_close is True


def test_no_exit_call_when_no_position_open(quotes_df, vix_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]
    market = _market_for(date, quotes_df)
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    strategy._signals["EURUSD"] = _fake_bundle(dates, exit_dates=(date,))
    portfolio = Portfolio()  # no open position at all

    assert strategy.generate_orders(date, market, portfolio) == []
