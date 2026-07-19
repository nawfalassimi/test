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
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=5).generate()


def _first_date_with_weekday(dates, weekday):
    return next(d for d in dates if d.weekday() == weekday)


def _market_for(date, quotes_df, pair="EURUSD"):
    return Market(date=date, snapshots={pair: build_market_snapshot(date, quotes_df, pair)})


def test_registered_under_expected_name():
    assert get_strategy("short_vol_carry_1m") is ShortVolCarryStrategy


def test_no_orders_on_non_monday(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    tuesday = _first_date_with_weekday(dates, 1)
    market = _market_for(tuesday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()

    assert strategy.generate_orders(tuesday, market, portfolio) == []


def test_monday_entry_generates_matching_call_put_clip(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    monday = _first_date_with_weekday(dates, 0)
    market = _market_for(monday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()

    orders = strategy.generate_orders(monday, market, portfolio)
    assert len(orders) == 2
    assert {o.instrument.option_type for o in orders} == {"call", "put"}
    assert all(o.side == "sell" for o in orders)
    assert all(o.strategy_id == "short_vol_carry_1m" for o in orders)
    clip_ids = {o.clip_id for o in orders}
    assert len(clip_ids) == 1
    assert all(o.instrument.strike == orders[0].instrument.strike for o in orders)


def test_no_new_entry_while_position_open(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    monday = _first_date_with_weekday(dates, 0)
    market = _market_for(monday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()
    portfolio.positions.append(Position(
        instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="short_vol_carry_1m_prev",
        strategy_id="short_vol_carry_1m", entry_date=monday, entry_price=0.0,
    ))

    assert strategy.generate_orders(monday, market, portfolio) == []


def _two_pair_market_for(monday, quotes_df):
    jpy_df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                      seed=50, base_spot=110.0).generate()
    return Market(date=monday, snapshots={
        "EURUSD": build_market_snapshot(monday, quotes_df, "EURUSD"),
        "USDJPY": build_market_snapshot(monday, jpy_df, "USDJPY"),
    })


def test_pairs_kwarg_sets_required_pairs():
    assert ShortVolCarryStrategy(pairs=["EURUSD", "USDJPY"]).required_pairs == ["EURUSD", "USDJPY"]


def test_legacy_pair_kwarg_still_sets_required_pairs():
    assert ShortVolCarryStrategy(pair="USDJPY").required_pairs == ["USDJPY"]
    assert ShortVolCarryStrategy().required_pairs == ["EURUSD"]


def test_duplicate_pairs_raises():
    with pytest.raises(ValueError):
        ShortVolCarryStrategy(pairs=["EURUSD", "EURUSD"])


def test_both_pair_and_pairs_raises():
    with pytest.raises(ValueError):
        ShortVolCarryStrategy(pair="USDJPY", pairs=["EURUSD"])


def test_monday_entry_generates_independent_clips_per_pair(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    monday = _first_date_with_weekday(dates, 0)
    market = _two_pair_market_for(monday, quotes_df)
    strategy = ShortVolCarryStrategy(pairs=["EURUSD", "USDJPY"])
    portfolio = Portfolio()

    orders = strategy.generate_orders(monday, market, portfolio)
    assert len(orders) == 4
    assert {o.instrument.pair for o in orders} == {"EURUSD", "USDJPY"}
    clip_ids = {o.clip_id for o in orders}
    assert len(clip_ids) == 2
    for pair in ("EURUSD", "USDJPY"):
        pair_clip_ids = {o.clip_id for o in orders if o.instrument.pair == pair}
        assert len(pair_clip_ids) == 1
        assert pair in next(iter(pair_clip_ids))


def test_open_position_in_one_pair_does_not_block_other_pair(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    monday = _first_date_with_weekday(dates, 0)
    market = _two_pair_market_for(monday, quotes_df)
    strategy = ShortVolCarryStrategy(pairs=["EURUSD", "USDJPY"])
    portfolio = Portfolio()
    portfolio.positions.append(Position(
        instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="short_vol_carry_1m_EURUSD_prev",
        strategy_id="short_vol_carry_1m", entry_date=monday, entry_price=0.0,
    ))

    orders = strategy.generate_orders(monday, market, portfolio)
    assert len(orders) == 2
    assert {o.instrument.pair for o in orders} == {"USDJPY"}
