from __future__ import annotations

import dataclasses
import math

import pytest

from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.market.snapshot import build_market_snapshot


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-03-31", seed=11).generate()


def test_snapshot_is_immutable(quotes_df):
    date = quotes_df["date"].iloc[0]
    snapshot = build_market_snapshot(date, quotes_df, "EURUSD")
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.spot = 999.0


def test_build_market_snapshot_is_deterministic(quotes_df):
    date = quotes_df["date"].iloc[20]
    snap_a = build_market_snapshot(date, quotes_df, "EURUSD")
    snap_b = build_market_snapshot(date, quotes_df, "EURUSD")

    assert snap_a.spot == snap_b.spot
    assert snap_a.r_d == snap_b.r_d
    assert snap_a.r_f == snap_b.r_f
    assert snap_a.forward(0.25) == snap_b.forward(0.25)
    assert snap_a.vol_for_delta(0.5, 0.25) == snap_b.vol_for_delta(0.5, 0.25)


def test_snapshot_pricing_helpers_are_self_consistent(quotes_df):
    date = quotes_df["date"].iloc[20]
    snapshot = build_market_snapshot(date, quotes_df, "EURUSD")

    T = 30 / 365
    K = snapshot.forward(T)
    vol_atm = snapshot.vol_for_delta(0.5, T)
    vol_for_atm_strike = snapshot.implied_vol_for_strike(K, T)
    # ATM-forward strike's implied vol should be close to the ATM (delta=0.5) vol.
    assert vol_for_atm_strike == pytest.approx(vol_atm, abs=5e-4)

    assert snapshot.discount_factor(T, domestic=True) == pytest.approx(math.exp(-snapshot.r_d * T))
