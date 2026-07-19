from __future__ import annotations

import math

from fxbacktest.data.synthetic import SyntheticFxDataGenerator, SyntheticVixGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_signal import ShortVolSignalStrategy

START, END = "2022-01-01", "2023-12-29"


def _run(hedge_mode: str):
    quotes_df = SyntheticFxDataGenerator(pair="EURUSD", start=START, end=END, seed=60).generate()
    vix_df = SyntheticVixGenerator(start=START, end=END, seed=61).generate()

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolSignalStrategy(quotes_df=quotes_df, vix_df=vix_df)
    hedger = DailyDeltaHedger(pricer, mode=hedge_mode)

    return run_backtest(quotes_df, [strategy], hedger, pricer, vix_df=vix_df)


def test_signal_strategy_runs_end_to_end_with_no_hedge():
    result_df, portfolio = _run("none")

    assert len(result_df) > 100
    assert result_df["pnl"].apply(math.isfinite).all()
    assert result_df["cum_pnl"].apply(math.isfinite).all()

    strategy_positions = [p for p in portfolio.positions if p.strategy_id == "short_vol_signal"]
    assert len(strategy_positions) > 0  # at least one entry occurred

    early_closes = [
        p for p in strategy_positions
        if not p.is_open and isinstance(p.instrument, FxVanillaOption) and p.exit_date < p.instrument.expiry
    ]
    assert len(early_closes) > 0  # a hard-stop early close occurred before natural expiry

    # no hedge trades at all in "none" mode
    assert not any(p.strategy_id == "hedge" for p in portfolio.positions)


def test_signal_strategy_runs_with_daily_and_threshold_hedge():
    for mode in ("daily", "threshold"):
        result_df, portfolio = _run(mode)
        assert len(result_df) > 100
        assert result_df["pnl"].apply(math.isfinite).all()
        assert result_df["cum_pnl"].apply(math.isfinite).all()
