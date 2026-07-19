from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.execution.order import Order
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=9).generate()


def _snapshot_with_open_straddle(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    snapshot = build_market_snapshot(dates[0], quotes_df)

    K = snapshot.forward(30 / 365)
    expiry = snapshot.date + pd.Timedelta(days=30)
    call = FxVanillaOption(pair="EURUSD", strike=K, expiry=expiry, option_type="call",
                           notional=1_000_000, trade_date=snapshot.date)
    put = FxVanillaOption(pair="EURUSD", strike=K, expiry=expiry, option_type="put",
                          notional=1_000_000, trade_date=snapshot.date)
    orders = [
        Order(instrument=call, side="sell", qty=1.0, clip_id="c1", strategy_id="short_vol_carry_1m"),
        Order(instrument=put, side="sell", qty=1.0, clip_id="c1", strategy_id="short_vol_carry_1m"),
    ]
    portfolio.execute(orders, snapshot, pricer)
    return snapshot, portfolio, pricer


def test_daily_hedge_flattens_net_delta(quotes_df):
    snapshot, portfolio, pricer = _snapshot_with_open_straddle(quotes_df)
    hedger = DailyDeltaHedger(pricer, mode="daily")

    delta_before = portfolio.net_delta(snapshot, pricer)
    orders = hedger.rehedge_orders(snapshot.date, snapshot, portfolio)
    assert len(orders) == 1

    portfolio.execute(orders, snapshot, pricer)
    delta_after = portfolio.net_delta(snapshot, pricer)

    assert abs(delta_after) < abs(delta_before)
    assert delta_after == pytest.approx(0.0, abs=1e-6)


def test_threshold_mode_skips_hedge_below_limit(quotes_df):
    snapshot, portfolio, pricer = _snapshot_with_open_straddle(quotes_df)
    delta_before = abs(portfolio.net_delta(snapshot, pricer))

    hedger = DailyDeltaHedger(pricer, mode="threshold", threshold=delta_before * 10)
    assert hedger.rehedge_orders(snapshot.date, snapshot, portfolio) == []


def test_threshold_mode_hedges_above_limit(quotes_df):
    snapshot, portfolio, pricer = _snapshot_with_open_straddle(quotes_df)
    delta_before = abs(portfolio.net_delta(snapshot, pricer))

    hedger = DailyDeltaHedger(pricer, mode="threshold", threshold=delta_before / 2)
    orders = hedger.rehedge_orders(snapshot.date, snapshot, portfolio)
    assert len(orders) == 1
