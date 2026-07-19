from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.market.market import Market
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-09-30", seed=12).generate()


def _market_for(date, quotes_df, pair="EURUSD"):
    return Market(date=date, snapshots={pair: build_market_snapshot(date, quotes_df, pair)})


def test_at_most_one_open_hedge_position_per_pair_after_each_day(quotes_df):
    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    result_df, portfolio = run_backtest(quotes_df, [strategy], hedger, pricer)

    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    last_date = dates[-1]

    hedge_positions = [pos for pos in portfolio.positions if pos.strategy_id == "hedge"]
    assert len(hedge_positions) > 0
    # blow-up guard: the pre-fix bug produced roughly len(dates)^2/2 rows
    assert len(hedge_positions) < 3 * len(dates)

    intervals = sorted(
        (pos.entry_date, pos.exit_date if not pos.is_open else last_date)
        for pos in hedge_positions
    )
    for (_, end_a), (start_b, _) in zip(intervals, intervals[1:]):
        assert start_b >= end_a, "two hedge positions were open at the same time"


def test_hedge_consolidation_is_nav_neutral(quotes_df):
    pricer = GarmanKohlhagenPricer()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market_day1 = _market_for(dates[0], quotes_df)
    market_day2 = _market_for(dates[1], quotes_df)

    def make_positions():
        return [
            Position(instrument=FxSpot(pair="EURUSD", notional=30_000.0), qty=1.0,
                    clip_id="hedge_a", strategy_id="hedge", entry_date=dates[0], entry_price=33_000.0),
            Position(instrument=FxSpot(pair="EURUSD", notional=10_000.0), qty=-1.0,
                    clip_id="hedge_b", strategy_id="hedge", entry_date=dates[0], entry_price=-11_000.0),
        ]

    consolidated = Portfolio(positions=make_positions())
    consolidated.mark_to_market(market_day1, pricer)
    consolidated.mark_to_market(market_day2, pricer)

    unconsolidated = Portfolio(positions=make_positions())
    unconsolidated._consolidate_hedge_positions = lambda *args, **kwargs: None  # disable, isolate the effect
    unconsolidated.mark_to_market(market_day1, pricer)
    unconsolidated.mark_to_market(market_day2, pricer)

    assert consolidated.cum_pnl == pytest.approx(unconsolidated.cum_pnl)
    assert len([p for p in consolidated.positions if p.is_open]) == 1
    assert len([p for p in unconsolidated.positions if p.is_open]) == 2


def test_blotter_row_count_stays_linear_with_hedge_included():
    from fxbacktest.analytics.blotter import build_trade_blotter

    df = SyntheticFxDataGenerator(start="2022-01-01", end="2022-12-30", seed=13).generate()
    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    result_df, portfolio = run_backtest(df, [strategy], hedger, pricer)

    n_dates = len(df["date"].drop_duplicates())
    blotter = build_trade_blotter(portfolio, df, pricer)

    assert (blotter["strategy_id"] == "hedge").any()
    assert len(blotter) < 20 * n_dates
