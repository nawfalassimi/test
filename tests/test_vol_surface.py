from __future__ import annotations

import pytest

from fxbacktest.data.schema import TENOR_YEARS
from fxbacktest.market.vol_surface import VolSurface


def _flat_quotes(atm_vol, rr25=-0.01, bf25=0.003, rr10=-0.015, bf10=0.005):
    return {"atm_vol": atm_vol, "rr25": rr25, "bf25": bf25, "rr10": rr10, "bf10": bf10}


def test_get_vol_reproduces_atm_input_at_pillars():
    tenor_quotes = {
        "1W": _flat_quotes(0.05), "1M": _flat_quotes(0.06), "3M": _flat_quotes(0.07),
        "6M": _flat_quotes(0.08), "1Y": _flat_quotes(0.09),
    }
    surface = VolSurface(tenor_quotes)
    for tenor, q in tenor_quotes.items():
        T = TENOR_YEARS[tenor]
        assert surface.get_vol(delta_call=0.50, T=T) == pytest.approx(q["atm_vol"], abs=1e-10)


def test_total_variance_monotonically_increasing_across_pillars():
    # Strictly increasing ATM vol term structure -> total variance vol^2*T must
    # also be strictly increasing across pillars (no calendar arbitrage).
    tenor_quotes = {
        "1W": _flat_quotes(0.05), "1M": _flat_quotes(0.06), "3M": _flat_quotes(0.07),
        "6M": _flat_quotes(0.08), "1Y": _flat_quotes(0.09),
    }
    surface = VolSurface(tenor_quotes)
    variances = [surface.get_vol(0.50, TENOR_YEARS[t]) ** 2 * TENOR_YEARS[t] for t in
                 ["1W", "1M", "3M", "6M", "1Y"]]
    assert all(v2 > v1 for v1, v2 in zip(variances, variances[1:]))


def test_interpolated_tenor_variance_bounded_by_neighboring_pillars():
    tenor_quotes = {
        "1W": _flat_quotes(0.05), "1M": _flat_quotes(0.06), "3M": _flat_quotes(0.07),
        "6M": _flat_quotes(0.08), "1Y": _flat_quotes(0.09),
    }
    surface = VolSurface(tenor_quotes)
    T_1m, T_3m = TENOR_YEARS["1M"], TENOR_YEARS["3M"]
    mid_T = (T_1m + T_3m) / 2

    w_1m = surface.get_vol(0.50, T_1m) ** 2 * T_1m
    w_3m = surface.get_vol(0.50, T_3m) ** 2 * T_3m
    w_mid = surface.get_vol(0.50, mid_T) ** 2 * mid_T

    assert min(w_1m, w_3m) <= w_mid <= max(w_1m, w_3m)


def test_get_vol_for_strike_round_trips_with_solve_strike_for_delta():
    from fxbacktest.market.smile import solve_strike_for_delta

    tenor_quotes = {
        "1W": _flat_quotes(0.05), "1M": _flat_quotes(0.06), "3M": _flat_quotes(0.07),
        "6M": _flat_quotes(0.08), "1Y": _flat_quotes(0.09),
    }
    surface = VolSurface(tenor_quotes)
    S, r_d, r_f, T = 1.10, 0.02, 0.01, TENOR_YEARS["3M"]

    vol_fn = lambda delta: surface.get_vol(delta, T)  # noqa: E731
    K, expected_vol = solve_strike_for_delta(0.30, S, r_d, r_f, T, vol_fn)

    vol_for_strike = surface.get_vol_for_strike(K, S, r_d, r_f, T)
    assert vol_for_strike == pytest.approx(expected_vol, abs=1e-6)


def test_get_vol_across_smile_reflects_skew():
    # With negative RR (puts richer than calls, typical downside-skew), the
    # 10-delta-put-equivalent vol (call-delta 0.90) should exceed the
    # 10-delta-call-equivalent vol (call-delta 0.10).
    tenor_quotes = {t: _flat_quotes(0.07, rr25=-0.02, bf25=0.004, rr10=-0.03, bf10=0.006)
                    for t in TENOR_YEARS}
    surface = VolSurface(tenor_quotes)
    T = TENOR_YEARS["3M"]
    vol_10p_equiv = surface.get_vol(0.90, T)
    vol_10c_equiv = surface.get_vol(0.10, T)
    assert vol_10p_equiv > vol_10c_equiv
