from __future__ import annotations

import math

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.execution.order import Order
from fxbacktest.execution.transaction_costs import OptionCostSpec, PairCostSpec, TransactionCostModel
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.market.market import Market
from fxbacktest.market.snapshot import build_market_snapshot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=3).generate()


def _market_for(date, quotes_df, pair="EURUSD"):
    return Market(date=date, snapshots={pair: build_market_snapshot(date, quotes_df, pair)})


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
    market0 = _market_for(dates[0], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    baseline = portfolio.mark_to_market(market0, pricer)
    assert baseline["pnl"] == pytest.approx(0.0)
    assert baseline["cum_pnl"] == pytest.approx(0.0)

    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model())

    assert len(portfolio.positions) == 2
    assert portfolio.has_open_position("short_vol_carry_1m")

    after_open = portfolio.mark_to_market(market0, pricer)
    assert after_open["pnl"] == pytest.approx(0.0, abs=1e-6)
    assert portfolio.cum_pnl == pytest.approx(0.0, abs=1e-6)


def test_transaction_costs_create_immediate_negative_pnl(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    portfolio.mark_to_market(market0, pricer)
    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model(vol_spread_bp=50.0))

    after_open = portfolio.mark_to_market(market0, pricer)
    # Selling at a worse (lower) vol than fair mid should show an immediate cost drag.
    assert after_open["pnl"] < 0
    assert portfolio.cum_pnl < 0


def test_mark_to_market_next_day_reflects_market_move(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    market1 = _market_for(dates[1], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    portfolio.mark_to_market(market0, pricer)
    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model())
    portfolio.mark_to_market(market0, pricer)

    next_day = portfolio.mark_to_market(market1, pricer)
    assert math.isfinite(next_day["pnl"])
    assert next_day["delta"] != 0 or next_day["gamma"] != 0


def test_cost_paid_is_zero_with_zero_cost_model(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model())

    assert all(pos.cost_paid == 0.0 for pos in portfolio.positions)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_cost_paid_matches_abs_adjustment_for_both_sides(quotes_df, side):
    pricer = GarmanKohlhagenPricer()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    T_days = 30
    K = snapshot0.forward(T_days / 365)
    expiry = snapshot0.date + pd.Timedelta(days=T_days)
    call = FxVanillaOption(pair="EURUSD", strike=K, expiry=expiry, option_type="call",
                           notional=1_000_000, trade_date=snapshot0.date)
    order = Order(instrument=call, side=side, qty=1.0, clip_id="c1", strategy_id="s")

    cost_model = _cost_model(vol_spread_bp=50.0)
    portfolio = Portfolio()
    portfolio.execute([order], market0, pricer, cost_model)

    # EURUSD's quote currency IS USD (rate 1.0), so the USD-converted values
    # below are numerically identical to the pre-multi-currency native ones —
    # this test still exercises real conversion logic (Portfolio.instrument_value/
    # instrument_greeks go through Market.usd_rate), just at a degenerate rate.
    fair_price = portfolio.instrument_value(call, market0, pricer)
    vega = portfolio.instrument_greeks(call, market0, pricer).vega
    expected_adjustment = cost_model.option_cost("EURUSD", side, fair_price, vega)

    assert portfolio.positions[0].cost_paid == pytest.approx(abs(expected_adjustment))
    assert portfolio.positions[0].cost_paid > 0.0


def test_mark_to_market_friction_cost_sums_same_day_entries(quotes_df):
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    market1 = _market_for(dates[1], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model(vol_spread_bp=50.0))

    entry_day = portfolio.mark_to_market(market0, pricer)
    expected_cost = sum(pos.cost_paid for pos in portfolio.positions)
    assert entry_day["friction_cost"] == pytest.approx(expected_cost)
    assert entry_day["friction_cost"] > 0.0

    next_day = portfolio.mark_to_market(market1, pricer)
    assert next_day["friction_cost"] == pytest.approx(0.0)


def test_instrument_value_converts_eurhuf_option_to_usd():
    """EURHUF's quote currency (HUF) isn't USD, so this exercises the one-hop
    cross-conversion via a EURUSD bridge — not the degenerate quote==USD case
    the EURUSD-only tests above all hit."""
    pricer = GarmanKohlhagenPricer()
    date = pd.Timestamp("2022-01-03")
    eurhuf_df = SyntheticFxDataGenerator(pair="EURHUF", start="2022-01-01", end="2022-03-31",
                                         seed=20, base_spot=400.0).generate()
    eurusd_df = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-03-31", seed=21).generate()
    eurhuf_snap = build_market_snapshot(date, eurhuf_df, "EURHUF")
    eurusd_snap = build_market_snapshot(date, eurusd_df, "EURUSD")
    market = Market(date=date, snapshots={"EURHUF": eurhuf_snap, "EURUSD": eurusd_snap})

    K = eurhuf_snap.forward(30 / 365)
    expiry = date + pd.Timedelta(days=30)
    option = FxVanillaOption(pair="EURHUF", strike=K, expiry=expiry, option_type="call",
                             notional=1_000_000, trade_date=date)

    portfolio = Portfolio()
    native_price = pricer.price(option, eurhuf_snap)  # HUF
    usd_price = portfolio.instrument_value(option, market, pricer)

    expected_usd_rate = eurusd_snap.spot / eurhuf_snap.spot  # USD per HUF, via the EUR bridge
    assert usd_price == pytest.approx(native_price * expected_usd_rate)

    native_greeks = pricer.greeks(option, eurhuf_snap)
    usd_greeks = portfolio.instrument_greeks(option, market, pricer)
    assert usd_greeks.delta == pytest.approx(native_greeks.delta * eurusd_snap.spot)  # base=EUR
    assert usd_greeks.vega == pytest.approx(native_greeks.vega * expected_usd_rate)  # quote=HUF


def test_instrument_value_converts_usdjpy_option_to_usd():
    """USDJPY's quote currency (JPY) isn't USD either, but its BASE currency
    IS USD — the inverse-rate branch of Market.usd_rate, no bridge needed."""
    pricer = GarmanKohlhagenPricer()
    date = pd.Timestamp("2022-01-03")
    df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                  seed=22, base_spot=110.0).generate()
    snap = build_market_snapshot(date, df, "USDJPY")
    market = Market(date=date, snapshots={"USDJPY": snap})

    K = snap.forward(30 / 365)
    expiry = date + pd.Timedelta(days=30)
    option = FxVanillaOption(pair="USDJPY", strike=K, expiry=expiry, option_type="put",
                             notional=1_000_000, trade_date=date)

    portfolio = Portfolio()
    native_price = pricer.price(option, snap)  # JPY
    usd_price = portfolio.instrument_value(option, market, pricer)
    assert usd_price == pytest.approx(native_price / snap.spot)  # USD per JPY = 1/spot(USDJPY)

    native_greeks = pricer.greeks(option, snap)
    usd_greeks = portfolio.instrument_greeks(option, market, pricer)
    assert usd_greeks.delta == pytest.approx(native_greeks.delta * 1.0)  # base=USD, rate 1.0
    assert usd_greeks.vega == pytest.approx(native_greeks.vega / snap.spot)  # quote=JPY


def test_has_open_position_pair_filter_matches_same_pair():
    portfolio = Portfolio(positions=[
        Position(instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="c1",
                strategy_id="s", entry_date=pd.Timestamp("2022-01-03"), entry_price=0.0),
    ])
    assert portfolio.has_open_position("s", pair="EURUSD")


def test_has_open_position_pair_filter_excludes_other_pair():
    """Regression test for the exact bug being fixed: an open EURUSD position
    must not read as open when queried for a different pair under the same
    strategy_id, even though the unfiltered call still reports it as open."""
    portfolio = Portfolio(positions=[
        Position(instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="c1",
                strategy_id="s", entry_date=pd.Timestamp("2022-01-03"), entry_price=0.0),
    ])
    assert not portfolio.has_open_position("s", pair="USDJPY")
    assert portfolio.has_open_position("s")  # unfiltered call keeps old behavior


def test_mark_for_early_close_defers_to_next_mark_to_market(quotes_df):
    """Regression test for the P&L-leak bug a naive eager close would cause:
    mark_for_early_close must only flag the position; the FULL last-mark-to-
    close move must still be accrued by the mark_to_market call that actually
    processes the flag, not silently dropped because is_open flips too early."""
    pricer = GarmanKohlhagenPricer()
    portfolio = Portfolio()
    dates = sorted(quotes_df["date"].drop_duplicates().tolist())
    market0 = _market_for(dates[0], quotes_df)
    market1 = _market_for(dates[1], quotes_df)
    snapshot0 = market0.snapshot("EURUSD")

    orders = _straddle_orders(snapshot0)
    portfolio.execute(orders, market0, pricer, _cost_model())
    portfolio.mark_to_market(market0, pricer)

    assert portfolio.has_open_position("short_vol_carry_1m")

    portfolio.mark_for_early_close("short_vol_carry_1m", "EURUSD")
    # Still open immediately after flagging — the close is lazy.
    assert portfolio.has_open_position("short_vol_carry_1m")

    last_mark_prices = {id(pos): pos.last_mark_price for pos in portfolio.positions}
    expected_values = {id(pos): portfolio.instrument_value(pos.instrument, market1, pricer)
                      for pos in portfolio.positions}
    expected_pnl = sum(pos.qty * (expected_values[id(pos)] - last_mark_prices[id(pos)])
                       for pos in portfolio.positions if pos.is_open)

    next_day = portfolio.mark_to_market(market1, pricer)
    assert next_day["pnl"] == pytest.approx(expected_pnl)
    assert not portfolio.has_open_position("short_vol_carry_1m")
    assert all(not pos.is_open and pos.exit_date == market1.date for pos in portfolio.positions)


def test_mark_for_early_close_only_matches_strategy_id_and_pair():
    portfolio = Portfolio(positions=[
        Position(instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="c1",
                strategy_id="s", entry_date=pd.Timestamp("2022-01-03"), entry_price=0.0),
        Position(instrument=FxSpot(pair="USDJPY", notional=0.0), qty=1.0, clip_id="c2",
                strategy_id="s", entry_date=pd.Timestamp("2022-01-03"), entry_price=0.0),
        Position(instrument=FxSpot(pair="EURUSD", notional=0.0), qty=1.0, clip_id="c3",
                strategy_id="other", entry_date=pd.Timestamp("2022-01-03"), entry_price=0.0),
    ])
    portfolio.mark_for_early_close("s", "EURUSD")

    flagged = {pos.clip_id for pos in portfolio.positions if pos.pending_close}
    assert flagged == {"c1"}


def test_native_delta_by_pair_groups_correctly_and_stays_native(quotes_df):
    pricer = GarmanKohlhagenPricer()
    date = sorted(quotes_df["date"].drop_duplicates().tolist())[0]
    eur_snap = build_market_snapshot(date, quotes_df, "EURUSD")
    jpy_df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-03-31",
                                      seed=30, base_spot=110.0).generate()
    jpy_snap = build_market_snapshot(date, jpy_df, "USDJPY")
    market = Market(date=date, snapshots={"EURUSD": eur_snap, "USDJPY": jpy_snap})

    portfolio = Portfolio(positions=[
        Position(instrument=FxSpot(pair="EURUSD", notional=100_000.0), qty=1.0,
                clip_id="c1", strategy_id="s", entry_date=date, entry_price=0.0),
        Position(instrument=FxSpot(pair="USDJPY", notional=50_000.0), qty=-1.0,
                clip_id="c2", strategy_id="s", entry_date=date, entry_price=0.0),
    ])

    natives = portfolio.native_delta_by_pair(market, pricer)
    # native (unconverted) deltas — NOT run through usd_rate, unlike net_delta
    assert natives == {"EURUSD": 100_000.0, "USDJPY": -50_000.0}
