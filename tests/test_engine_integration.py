from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fxbacktest.data.schema import QUOTE_COLUMNS, TENOR_YEARS
from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.execution.transaction_costs import OptionCostSpec, PairCostSpec, SpotCostSpec, TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy


def _frozen_vol_carry_quotes(seed: int, start: str, end: str) -> pd.DataFrame:
    """Quotes with a frozen (non-noisy) vol surface AND frozen carry, so only
    spot and calendar time vary. This isolates the pure delta-hedged
    gamma/theta relationship: with vol and rates held constant, no vega or
    rho-like P&L should leak into the identity check below."""
    gen = SyntheticFxDataGenerator(start=start, end=end, seed=seed)
    spot = gen.generate_spot_path()
    carry = pd.Series(0.015, index=gen.dates)
    fwd_df = gen.generate_forward_points(spot, carry)

    vol_rows = [
        pd.DataFrame({"date": gen.dates, "tenor": tenor, "atm_vol": 0.08,
                     "rr25": -0.01, "bf25": 0.003, "rr10": -0.015, "bf10": 0.005})
        for tenor in TENOR_YEARS
    ]
    vol_df = pd.concat(vol_rows, ignore_index=True)

    df = fwd_df.merge(vol_df, on=["date", "tenor"])
    df["spot"] = df["date"].map(spot)
    df["pair"] = "EURUSD"
    return df[["pair"] + QUOTE_COLUMNS].sort_values(["date", "tenor"]).reset_index(drop=True), spot


def test_daily_hedged_pnl_tracks_gamma_theta_identity():
    """README's core sanity check: daily hedged P&L on the short straddle should
    roughly track 0.5*gamma*(dS)^2 + theta*dt (this codebase's theta convention
    is dV/d(calendar time), so the theta term is added, not subtracted — see
    GarmanKohlhagenPricer). Checked statistically over the sample, not per-day,
    since this is a 2nd-order Taylor approximation under discrete daily hedging.
    """
    df, spot = _frozen_vol_carry_quotes(seed=1, start="2022-01-01", end="2022-12-30")

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    result, _portfolio = run_backtest(df, [strategy], hedger, pricer)

    result["spot"] = result["date"].map(spot)
    result["dS"] = result["spot"].diff()
    result["dt"] = result["date"].diff().dt.days / 365.0
    result["gamma_lag"] = result["gamma"].shift(1)
    result["theta_lag"] = result["theta"].shift(1)
    result["predicted"] = 0.5 * result["gamma_lag"] * result["dS"] ** 2 + result["theta_lag"] * result["dt"]

    sub = result.dropna()
    sub = sub[sub["gamma_lag"] != 0]  # only days with an open, hedged position
    assert len(sub) > 100

    # Exclude the last day or two before expiry: theta/gamma blow up as T->0,
    # where a 2nd-order Taylor approximation over a full-day step breaks down
    # (a well-known limitation, not specific to this implementation).
    thresh = sub["theta_lag"].abs().quantile(0.90)
    trimmed = sub[sub["theta_lag"].abs() <= thresh]

    residual = trimmed["pnl"] - trimmed["predicted"]
    mean_abs_pnl = trimmed["pnl"].abs().mean()

    assert abs(residual.mean()) < 0.5 * mean_abs_pnl
    assert abs(np.corrcoef(residual, trimmed["dS"] ** 2)[0, 1]) < 0.4
    assert np.corrcoef(trimmed["pnl"], trimmed["predicted"])[0, 1] > 0.5


def test_full_backtest_runs_end_to_end_with_costs_and_noise():
    """Acceptance check for the milestone: the full loop, with the noisy
    synthetic surface and realistic transaction costs, runs over ~2 years
    without exceptions and produces a bounded, finite P&L series."""
    gen = SyntheticFxDataGenerator(start="2022-01-01", end="2023-12-29", seed=42)
    df = gen.generate()

    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    cost_model = TransactionCostModel(by_pair={
        "EURUSD": PairCostSpec(option=OptionCostSpec(kind="vol_spread", vol_spread_bp=50.0),
                              spot=SpotCostSpec(spread_pips=1.0)),
    })
    result, portfolio = run_backtest(df, [strategy], hedger, pricer, assumed_foreign_rate=0.0, cost_model=cost_model)

    assert len(result) > 400
    assert result["pnl"].apply(math.isfinite).all()
    assert result["cum_pnl"].apply(math.isfinite).all()

    notional = 1_000_000
    assert result["cum_pnl"].abs().max() < notional  # no runaway blow-up

    assert len(portfolio.positions) > 0
    assert all(pos.cost_paid >= 0 for pos in portfolio.positions)
    assert result["friction_cost"].sum() == pytest.approx(
        sum(pos.cost_paid for pos in portfolio.positions)
    )
