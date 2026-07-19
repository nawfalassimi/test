from __future__ import annotations

import pytest

from fxbacktest.market.smile import build_smile_points, interpolate_smile


def test_build_smile_points_matches_formula():
    atm_vol, rr25, bf25, rr10, bf10 = 0.08, -0.015, 0.003, -0.022, 0.005
    points = build_smile_points(atm_vol, rr25, bf25, rr10, bf10)

    by_delta = {p.delta: p.vol for p in points}
    assert by_delta[0.50] == pytest.approx(atm_vol)
    assert by_delta[0.25] == pytest.approx(atm_vol + bf25 + rr25 / 2)
    assert by_delta[0.75] == pytest.approx(atm_vol + bf25 - rr25 / 2)
    assert by_delta[0.10] == pytest.approx(atm_vol + bf10 + rr10 / 2)
    assert by_delta[0.90] == pytest.approx(atm_vol + bf10 - rr10 / 2)

    deltas = [p.delta for p in points]
    assert deltas == sorted(deltas)


def test_interpolate_smile_reproduces_input_nodes():
    atm_vol, rr25, bf25, rr10, bf10 = 0.075, 0.012, 0.0025, 0.019, 0.0045
    points = build_smile_points(atm_vol, rr25, bf25, rr10, bf10)
    smile_fn = interpolate_smile(points)

    for point in points:
        assert smile_fn(point.delta) == pytest.approx(point.vol, abs=1e-10)


def test_interpolate_smile_is_smooth_between_nodes():
    points = build_smile_points(atm_vol=0.08, rr25=-0.01, bf25=0.003, rr10=-0.015, bf10=0.005)
    smile_fn = interpolate_smile(points)

    # Vol at a point strictly between two nodes should lie in a plausible range,
    # not wildly overshoot due to spline oscillation.
    vols = [p.vol for p in points]
    mid_vol = smile_fn(0.35)  # between 25C (0.25) and ATM (0.50)
    assert min(vols) - 0.02 <= mid_vol <= max(vols) + 0.02
