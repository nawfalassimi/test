from __future__ import annotations

import pandas as pd

from fxbacktest.analytics.trades import extract_trade_events
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position

ENTRY = pd.Timestamp("2022-01-03")
EXIT = pd.Timestamp("2022-02-02")


def _option_leg(option_type, **overrides):
    defaults = dict(
        instrument=FxVanillaOption(pair="EURUSD", strike=1.10, expiry=EXIT, option_type=option_type,
                                   notional=1_000_000, trade_date=ENTRY),
        qty=-1.0, clip_id="short_vol_carry_1m_20220103", strategy_id="short_vol_carry_1m",
        entry_date=ENTRY, entry_price=10_000.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def _hedge_leg(**overrides):
    defaults = dict(
        instrument=FxSpot(pair="EURUSD", notional=50_000.0),
        qty=1.0, clip_id="hedge_20220103", strategy_id="hedge",
        entry_date=ENTRY, entry_price=55_000.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_hedge_clips_excluded_by_default():
    portfolio = Portfolio(positions=[
        _option_leg("call", is_open=False, exit_date=EXIT),
        _option_leg("put", is_open=False, exit_date=EXIT),
        _hedge_leg(),
    ])
    events = extract_trade_events(portfolio)
    assert len(events) == 1
    assert events.iloc[0]["strategy_id"] == "short_vol_carry_1m"


def test_call_put_legs_paired_into_one_closed_event():
    portfolio = Portfolio(positions=[
        _option_leg("call", is_open=False, exit_date=EXIT),
        _option_leg("put", is_open=False, exit_date=EXIT),
    ])
    events = extract_trade_events(portfolio)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["clip_id"] == "short_vol_carry_1m_20220103"
    assert row["entry_date"] == ENTRY
    assert row["exit_date"] == EXIT
    assert row["is_closed"] == True


def test_still_open_clip_has_none_exit_date():
    portfolio = Portfolio(positions=[
        _option_leg("call", is_open=True),
        _option_leg("put", is_open=True),
    ])
    events = extract_trade_events(portfolio)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["exit_date"] is None
    assert row["is_closed"] == False


def test_exclude_strategy_ids_is_overridable():
    portfolio = Portfolio(positions=[_hedge_leg()])
    events = extract_trade_events(portfolio, exclude_strategy_ids=())
    assert len(events) == 1
    assert events.iloc[0]["strategy_id"] == "hedge"
