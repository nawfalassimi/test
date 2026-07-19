from __future__ import annotations

from typing import NamedTuple, Optional

import pandas as pd

from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.pricing.base import Pricer
from fxbacktest.strategies.base import Strategy


class BacktestResult(NamedTuple):
    result_df: pd.DataFrame
    portfolio: Portfolio

# Execution timing: same-day-close. Orders are generated from the date-T
# snapshot and fill at date-T close — no intraday/next-close lag in v1. This
# is a simplification to revisit later, not a look-ahead bug: MarketSnapshot
# is immutable and built once per date, so signal generation and fill pricing
# see exactly the same (already-known, close-of-day) market data.
#
# No risk-engine gate in this milestone (README's risk limits engine is
# explicitly out of scope for milestone 1) — orders are executed unconditionally.
#
# Ordering within a day: strategy orders execute BEFORE the hedge is computed,
# so a position opened today is hedged today (not left one day unhedged).
# The portfolio is then mark_to_market'd once, at the end of the day, after
# all of that day's trades — this is the end-of-day close snapshot the daily
# record (pnl, delta, gamma, ...) reflects.


def run_backtest(quotes_df: pd.DataFrame, strategy: Strategy, hedger: DailyDeltaHedger,
                  pricer: Pricer, assumed_foreign_rate: float = 0.0,
                  cost_model: Optional[TransactionCostModel] = None) -> BacktestResult:
    cost_model = cost_model or TransactionCostModel.zero()
    portfolio = Portfolio()
    dates = pd.DatetimeIndex(sorted(quotes_df["date"].drop_duplicates().tolist()))

    records = []
    for date in dates:
        snapshot = build_market_snapshot(date, quotes_df, assumed_foreign_rate)

        strategy_orders = strategy.generate_orders(date, snapshot, portfolio)
        portfolio.execute(strategy_orders, snapshot, pricer, cost_model)

        hedge_orders = hedger.rehedge_orders(date, snapshot, portfolio)
        portfolio.execute(hedge_orders, snapshot, pricer, cost_model)

        daily_metrics = portfolio.mark_to_market(snapshot, pricer)
        records.append(daily_metrics)

    return BacktestResult(pd.DataFrame(records), portfolio)
