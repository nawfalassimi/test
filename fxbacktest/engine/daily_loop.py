from __future__ import annotations

from typing import List, NamedTuple, Optional

import pandas as pd

from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.market.market import Market, bridge_pairs_for
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.pricing.base import Pricer
from fxbacktest.strategies.base import Strategy


class BacktestResult(NamedTuple):
    result_df: pd.DataFrame
    portfolio: Portfolio

# Execution timing: same-day-close. Orders are generated from the date-T
# market and fill at date-T close — no intraday/next-close lag in v1. This
# is a simplification to revisit later, not a look-ahead bug: Market wraps
# immutable MarketSnapshots built once per date, so signal generation and
# fill pricing see exactly the same (already-known, close-of-day) market data.
#
# No risk-engine gate in this milestone (README's risk limits engine is
# explicitly out of scope for milestone 1) — orders are executed unconditionally.
#
# Ordering within a day: strategy orders execute BEFORE the hedge is computed,
# so a position opened today is hedged today (not left one day unhedged).
# The portfolio is then mark_to_market'd once, at the end of the day, after
# all of that day's trades — this is the end-of-day close snapshot the daily
# record (pnl, delta, gamma, ...) reflects.
#
# Multi-currency: each strategy declares the pairs it needs via
# required_pairs (a single instance may require more than one pair). The
# market only loads the union of every strategy's required_pairs, plus any
# USD-conversion bridge pairs required for a traded pair that doesn't involve
# USD directly (see market.bridge_pairs_for) — so a EURUSD + USDJPY backtest
# never touches EURGBP data, for example.


def run_backtest(quotes_df: pd.DataFrame, strategies: List[Strategy], hedger: DailyDeltaHedger,
                  pricer: Pricer, assumed_foreign_rate: float = 0.0,
                  cost_model: Optional[TransactionCostModel] = None) -> BacktestResult:
    cost_model = cost_model or TransactionCostModel.zero()
    portfolio = Portfolio()
    dates = pd.DatetimeIndex(sorted(quotes_df["date"].drop_duplicates().tolist()))

    traded_pairs = {p for s in strategies for p in s.required_pairs}
    required_pairs = traded_pairs | bridge_pairs_for(traded_pairs)

    records = []
    for date in dates:
        snapshots = {pair: build_market_snapshot(date, quotes_df, pair, assumed_foreign_rate)
                    for pair in required_pairs}
        market = Market(date=date, snapshots=snapshots)

        strategy_orders = [order for strategy in strategies
                           for order in strategy.generate_orders(date, market, portfolio)]
        portfolio.execute(strategy_orders, market, pricer, cost_model)

        hedge_orders = hedger.rehedge_orders(date, market, portfolio)
        portfolio.execute(hedge_orders, market, pricer, cost_model)

        daily_metrics = portfolio.mark_to_market(market, pricer)
        records.append(daily_metrics)

    return BacktestResult(pd.DataFrame(records), portfolio)
