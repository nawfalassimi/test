from __future__ import annotations

import math

import pandas as pd
import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy

NOTIONAL = 1_000_000


def test_multi_currency_backtest_eurusd_and_usdjpy():
    """Two strategies, each trading its own single pair, run together —
    confirms the market only loads what's needed, hedges stay pair-isolated,
    and aggregate P&L/risk are finite, plausible USD magnitudes."""
    eurusd_df = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-06-30", seed=40).generate()
    usdjpy_df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-06-30",
                                        seed=41, base_spot=110.0).generate()
    combined = pd.concat([eurusd_df, usdjpy_df], ignore_index=True)

    pricer = GarmanKohlhagenPricer()
    eur_strategy = ShortVolCarryStrategy(pair="EURUSD", strategy_id="short_vol_carry_1m_eur")
    jpy_strategy = ShortVolCarryStrategy(pair="USDJPY", strategy_id="short_vol_carry_1m_jpy")
    hedger = DailyDeltaHedger(pricer, mode="daily")

    result_df, portfolio = run_backtest(combined, [eur_strategy, jpy_strategy], hedger, pricer)

    assert len(result_df) > 100
    assert result_df["pnl"].apply(math.isfinite).all()
    assert result_df["delta"].apply(math.isfinite).all()
    # bounded, plausible USD magnitude — a unit-conversion bug (e.g. treating
    # JPY amounts as if they were already USD) would blow this up by ~100x.
    assert result_df["cum_pnl"].abs().max() < 5 * NOTIONAL

    traded_pairs = {pos.instrument.pair for pos in portfolio.positions if pos.strategy_id != "hedge"}
    assert traded_pairs == {"EURUSD", "USDJPY"}

    hedge_pairs = {pos.instrument.pair for pos in portfolio.positions if pos.strategy_id == "hedge"}
    assert hedge_pairs <= {"EURUSD", "USDJPY"}  # no cross-contamination into some other pair


def test_multi_currency_single_strategy_instance_trades_both_pairs():
    """A single ShortVolCarryStrategy instance configured with pairs= (not two
    separate strategy instances) must trade both pairs — exercises required_pairs
    driving run_backtest's pair-loading for one instance's own multi-pair list."""
    eurusd_df = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-06-30", seed=44).generate()
    usdjpy_df = SyntheticFxDataGenerator(pair="USDJPY", start="2022-01-01", end="2022-06-30",
                                        seed=45, base_spot=110.0).generate()
    combined = pd.concat([eurusd_df, usdjpy_df], ignore_index=True)

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy(pairs=["EURUSD", "USDJPY"])
    hedger = DailyDeltaHedger(pricer, mode="daily")

    result_df, portfolio = run_backtest(combined, [strategy], hedger, pricer)

    assert len(result_df) > 100
    assert result_df["pnl"].apply(math.isfinite).all()
    assert result_df["cum_pnl"].abs().max() < 5 * NOTIONAL

    traded_pairs = {pos.instrument.pair for pos in portfolio.positions if pos.strategy_id != "hedge"}
    assert traded_pairs == {"EURUSD", "USDJPY"}


def test_required_pairs_drives_bridge_loading_for_single_multi_pair_instance():
    """A single instance whose required_pairs is ["EURHUF"] must still trigger
    the EURUSD bridge-loading path — this is a new code path through the
    changed daily_loop.py line (required_pairs contributed by one instance),
    distinct from the existing bridge test which uses pair= directly."""
    eurusd_df = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-06-30", seed=46).generate()
    eurhuf_df = SyntheticFxDataGenerator(pair="EURHUF", start="2022-01-01", end="2022-06-30",
                                         seed=47, base_spot=400.0).generate()
    combined = pd.concat([eurusd_df, eurhuf_df], ignore_index=True)

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy(pairs=["EURHUF"])
    hedger = DailyDeltaHedger(pricer, mode="daily")

    result_df, portfolio = run_backtest(combined, [strategy], hedger, pricer)

    assert len(result_df) > 100
    assert result_df["pnl"].apply(math.isfinite).all()

    traded_pairs = {pos.instrument.pair for pos in portfolio.positions}
    assert "EURHUF" in traded_pairs
    assert "EURUSD" not in traded_pairs  # bridge only, never traded


def test_multi_currency_backtest_with_eurhuf_bridge():
    """A strategy trading EURHUF only (neither leg is USD) must still produce
    a finite, USD-denominated result — this only works if the EURUSD bridge
    pair got auto-loaded and Market.usd_rate's one-hop cross kicked in; if the
    bridge weren't loaded, usd_rate("HUF") would raise and the backtest
    would blow up entirely."""
    eurusd_df = SyntheticFxDataGenerator(pair="EURUSD", start="2022-01-01", end="2022-06-30", seed=42).generate()
    eurhuf_df = SyntheticFxDataGenerator(pair="EURHUF", start="2022-01-01", end="2022-06-30",
                                         seed=43, base_spot=400.0).generate()
    combined = pd.concat([eurusd_df, eurhuf_df], ignore_index=True)

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy(pair="EURHUF")
    hedger = DailyDeltaHedger(pricer, mode="daily")

    result_df, portfolio = run_backtest(combined, [strategy], hedger, pricer)

    assert len(result_df) > 100
    assert result_df["pnl"].apply(math.isfinite).all()
    assert result_df["cum_pnl"].abs().max() < 5 * NOTIONAL

    traded_pairs = {pos.instrument.pair for pos in portfolio.positions}
    assert "EURHUF" in traded_pairs
    # EURUSD is only a conversion bridge here — the strategy never trades it
    assert "EURUSD" not in traded_pairs
