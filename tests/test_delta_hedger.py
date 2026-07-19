from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.execution.order import Order
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.market.market import Market
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=9).generate()


def _market_for(date, quotes_df, pair="EURUSD"):
    return Market(date=date, snapshots={pair: build_market_snapshot(date, quotes_df, pair)})


def _market_with_open_straddle(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market = _market_for(dates[0], quotes_df)
    snapshot = market.snapshot("EURUSD")

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
    portfolio.execute(orders, market, pricer)
    return market, portfolio, pricer


def test_daily_hedge_flattens_net_delta(quotes_df):
    market, portfolio, pricer = _market_with_open_straddle(quotes_df)
    hedger = DailyDeltaHedger(pricer, mode="daily")

    delta_before = portfolio.net_delta(market, pricer)
    orders = hedger.rehedge_orders(market.date, market, portfolio)
    assert len(orders) == 1

    portfolio.execute(orders, market, pricer)
    delta_after = portfolio.net_delta(market, pricer)

    assert abs(delta_after) < abs(delta_before)
    assert delta_after == pytest.approx(0.0, abs=1e-6)


def test_none_mode_never_hedges(quotes_df):
    market, portfolio, pricer = _market_with_open_straddle(quotes_df)
    delta_before = abs(portfolio.net_delta(market, pricer))
    assert delta_before > 0  # a genuinely open, unhedged straddle has nonzero delta

    hedger = DailyDeltaHedger(pricer, mode="none")
    assert hedger.rehedge_orders(market.date, market, portfolio) == []


def test_threshold_mode_skips_hedge_below_limit(quotes_df):
    market, portfolio, pricer = _market_with_open_straddle(quotes_df)
    delta_before = abs(portfolio.net_delta(market, pricer))

    hedger = DailyDeltaHedger(pricer, mode="threshold", threshold=delta_before * 10)
    assert hedger.rehedge_orders(market.date, market, portfolio) == []


def test_threshold_mode_hedges_above_limit(quotes_df):
    market, portfolio, pricer = _market_with_open_straddle(quotes_df)
    delta_before = abs(portfolio.net_delta(market, pricer))

    hedger = DailyDeltaHedger(pricer, mode="threshold", threshold=delta_before / 2)
    orders = hedger.rehedge_orders(market.date, market, portfolio)
    assert len(orders) == 1


def test_hedges_multiple_pairs_independently_in_native_currency(quotes_df):
    """A EURUSD straddle and a USDJPY straddle open at once must each get
    their own hedge order, sized in their OWN base currency — you cannot
    flatten EUR risk with a JPY-denominated trade."""
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    date = dates[0]

    eur_snap = build_market_snapshot(date, quotes_df, "EURUSD")
    jpy_df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                      seed=15, base_spot=110.0).generate()
    jpy_snap = build_market_snapshot(date, jpy_df, "USDJPY")
    market = Market(date=date, snapshots={"EURUSD": eur_snap, "USDJPY": jpy_snap})

    for pair, snap in (("EURUSD", eur_snap), ("USDJPY", jpy_snap)):
        K = snap.forward(30 / 365)
        expiry = date + pd.Timedelta(days=30)
        call = FxVanillaOption(pair=pair, strike=K, expiry=expiry, option_type="call",
                               notional=1_000_000, trade_date=date)
        put = FxVanillaOption(pair=pair, strike=K, expiry=expiry, option_type="put",
                              notional=1_000_000, trade_date=date)
        portfolio.execute([
            Order(instrument=call, side="sell", qty=1.0, clip_id=f"c_{pair}", strategy_id="s"),
            Order(instrument=put, side="sell", qty=1.0, clip_id=f"c_{pair}", strategy_id="s"),
        ], market, pricer)

    hedger = DailyDeltaHedger(pricer, mode="daily")
    orders = hedger.rehedge_orders(date, market, portfolio)

    assert len(orders) == 2
    orders_by_pair = {o.instrument.pair: o for o in orders}
    assert set(orders_by_pair) == {"EURUSD", "USDJPY"}

    natives = portfolio.native_delta_by_pair(market, pricer)
    for pair, order in orders_by_pair.items():
        expected_side = "sell" if natives[pair] > 0 else "buy"
        assert order.side == expected_side
        assert order.instrument.notional == pytest.approx(abs(natives[pair]))
        assert order.instrument.pair == pair  # no cross-contamination

    portfolio.execute(orders, market, pricer)
    for pair in ("EURUSD", "USDJPY"):
        assert portfolio.native_delta_by_pair(market, pricer)[pair] == pytest.approx(0.0, abs=1e-6)
