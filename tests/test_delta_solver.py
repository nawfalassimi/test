from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from fxbacktest.market.smile import (
    build_smile_points,
    interpolate_smile,
    solve_delta_for_strike,
    solve_strike_for_delta,
)

S, R_D, R_F, T = 1.20, 0.03, 0.01, 0.5


def _skewed_smile_fn():
    points = build_smile_points(atm_vol=0.08, rr25=-0.015, bf25=0.003, rr10=-0.025, bf10=0.005)
    return interpolate_smile(points)


def _gk_call_delta(K, vol):
    d1 = (math.log(S / K) + (R_D - R_F + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    return math.exp(-R_F * T) * norm.cdf(d1)


@pytest.mark.parametrize("target_delta", [0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90])
def test_solve_strike_for_delta_round_trips(target_delta):
    vol_fn = _skewed_smile_fn()
    K, vol = solve_strike_for_delta(target_delta, S, R_D, R_F, T, vol_fn)

    # Plugging K back through the GK call-delta formula (with the solved vol)
    # should reproduce the target delta.
    assert _gk_call_delta(K, vol) == pytest.approx(target_delta, abs=1e-6)


def test_solve_strike_for_delta_flat_smile_matches_closed_form():
    flat_vol = 0.10
    vol_fn = lambda delta: flat_vol  # noqa: E731

    for target_delta in [0.10, 0.25, 0.50, 0.75, 0.90]:
        K, vol = solve_strike_for_delta(target_delta, S, R_D, R_F, T, vol_fn)
        assert vol == pytest.approx(flat_vol)

        dd1 = norm.ppf(target_delta * math.exp(R_F * T))
        expected_K = S * math.exp(-dd1 * flat_vol * math.sqrt(T) + (R_D - R_F + 0.5 * flat_vol**2) * T)
        assert K == pytest.approx(expected_K, rel=1e-9)
        assert _gk_call_delta(K, flat_vol) == pytest.approx(target_delta, abs=1e-9)


@pytest.mark.parametrize("K", [1.05, 1.12, 1.20, 1.28, 1.38])
def test_solve_delta_for_strike_converges_and_round_trips(K):
    vol_fn = _skewed_smile_fn()
    delta, vol = solve_delta_for_strike(K, S, R_D, R_F, T, vol_fn)

    # Solving back from (delta, vol) to a strike should recover K.
    K_recovered, vol_recovered = solve_strike_for_delta(delta, S, R_D, R_F, T, vol_fn)
    assert K_recovered == pytest.approx(K, rel=1e-4)
    assert vol_recovered == pytest.approx(vol, abs=1e-8)


def test_solve_delta_for_strike_raises_when_not_converged():
    vol_fn = _skewed_smile_fn()
    with pytest.raises(Exception):
        solve_delta_for_strike(1.20, S, R_D, R_F, T, vol_fn, max_iter=0)
