from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.instruments.spot import FxSpot
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


def test_registered_under_expected_name():
    assert get_strategy("short_vol_carry_1m") is ShortVolCarryStrategy


def test_no_orders_on_non_monday(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    tuesday = _first_date_with_weekday(dates, 1)
    snapshot = build_market_snapshot(tuesday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()

    assert strategy.generate_orders(tuesday, snapshot, portfolio) == []


def test_monday_entry_generates_matching_call_put_clip(quotes_df):
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    monday = _first_date_with_weekday(dates, 0)
    snapshot = build_market_snapshot(monday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()

    orders = strategy.generate_orders(monday, snapshot, portfolio)
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
    snapshot = build_market_snapshot(monday, quotes_df)
    strategy = ShortVolCarryStrategy()
    portfolio = Portfolio()
    portfolio.positions.append(Position(
        instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="short_vol_carry_1m_prev",
        strategy_id="short_vol_carry_1m", entry_date=monday, entry_price=0.0,
    ))

    assert strategy.generate_orders(monday, snapshot, portfolio) == []
