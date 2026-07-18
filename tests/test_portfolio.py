from __future__ import annotations

import math

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.execution.order import Order
from fxbacktest.execution.transaction_costs import OptionCostSpec, PairCostSpec, TransactionCostModel
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=3).generate()


def _straddle_orders(snapshot, strategy_id="short_vol_carry_1m"):
    T_days = 30
    K = snapshot.forward(T_days / 365)
    expiry = snapshot.date + pd.Timedelta(days=T_days)
    clip_id = f"{strategy_id}_{snapshot.date:%Y%m%d}"
    call = FxVanillaOption(pair="EURUSD", strike=K, expiry=expiry, option_type="call",
                           notional=1_000_000, trade_date=snapshot.date)
    put = FxVanillaOption(pair="EURUSD", strike=K, expiry=expiry, option_type="put",
                          notional=1_000_000, trade_date=snapshot.date)
    return [
        Order(instrument=call, side="sell", qty=1.0, clip_id=clip_id, strategy_id=strategy_id),
        Order(instrument=put, side="sell", qty=1.0, clip_id=clip_id, strategy_id=strategy_id),
    ]


def _cost_model(vol_spread_bp=0.0, spot_spread_pips=0.0):
    return TransactionCostModel(by_pair={
        "EURUSD": PairCostSpec(option=OptionCostSpec(kind="vol_spread", vol_spread_bp=vol_spread_bp)),
    })


def test_portfolio_starts_with_zero_pnl_and_no_cash():
    portfolio = Portfolio()
    assert portfolio.cum_pnl == 0.0
    assert not hasattr(portfolio, "cash")


def test_selling_straddle_at_fair_value_creates_no_immediate_pnl(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    snapshot0 = build_market_snapshot(dates[0], quotes_df)

    baseline = portfolio.mark_to_market(snapshot0, pricer)
    assert baseline["pnl"] == pytest.approx(0.0)
    assert baseline["cum_pnl"] == pytest.approx(0.0)

    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, snapshot0, pricer, _cost_model())

    assert len(portfolio.positions) == 2
    assert portfolio.has_open_position("short_vol_carry_1m")

    after_open = portfolio.mark_to_market(snapshot0, pricer)
    assert after_open["pnl"] == pytest.approx(0.0, abs=1e-6)
    assert portfolio.cum_pnl == pytest.approx(0.0, abs=1e-6)


def test_transaction_costs_create_immediate_negative_pnl(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    snapshot0 = build_market_snapshot(dates[0], quotes_df)

    portfolio.mark_to_market(snapshot0, pricer)
    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, snapshot0, pricer, _cost_model(vol_spread_bp=50.0))

    after_open = portfolio.mark_to_market(snapshot0, pricer)
    # Selling at a worse (lower) vol than fair mid should show an immediate cost drag.
    assert after_open["pnl"] < 0
    assert portfolio.cum_pnl < 0


def test_mark_to_market_next_day_reflects_market_move(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    snapshot0 = build_market_snapshot(dates[0], quotes_df)
    snapshot1 = build_market_snapshot(dates[1], quotes_df)

    portfolio.mark_to_market(snapshot0, pricer)
    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, snapshot0, pricer, _cost_model())
    portfolio.mark_to_market(snapshot0, pricer)

    next_day = portfolio.mark_to_market(snapshot1, pricer)
    assert math.isfinite(next_day["pnl"])
    assert next_day["delta"] != 0 or next_day["gamma"] != 0
