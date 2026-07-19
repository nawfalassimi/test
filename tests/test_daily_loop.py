from __future__ import annotations

import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=4).generate()


def test_new_entry_is_hedged_same_day_not_next_day(quotes_df):
    """A straddle opened on entry day (a Monday) must be delta-hedged by the
    close of that SAME day — not left unhedged until the following day."""
    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    result, _portfolio = run_backtest(quotes_df, [strategy], hedger, pricer)

    mondays = result[result["date"].dt.weekday == 0]
    assert len(mondays) > 0

    entry_day = mondays.iloc[0]
    # The recorded "delta" reflects the end-of-day book, i.e. after the new
    # straddle was opened AND after that same day's hedge trade.
    assert entry_day["delta"] == pytest.approx(0.0, abs=1e-6)
    assert entry_day["gamma"] != 0  # confirms a position really was opened


def test_daily_records_reflect_end_of_day_hedged_state(quotes_df):
    """Every day with an open position should show ~0 net delta in the daily
    record, since the portfolio is mark_to_market'd after that day's hedge."""
    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    result, _portfolio = run_backtest(quotes_df, [strategy], hedger, pricer)

    with_position = result[result["gamma"] != 0]
    assert len(with_position) > 10
    assert with_position["delta"].abs().max() < 1e-6
