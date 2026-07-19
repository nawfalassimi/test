from __future__ import annotations

import math

import pandas as pd
import pytest

from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from tests.conftest import FakeSnapshot

PRICER = GarmanKohlhagenPricer()
TRADE_DATE = pd.Timestamp("2023-01-01")


def _make_option(K, T_years, option_type, notional=1.0):
    expiry = TRADE_DATE + pd.Timedelta(days=round(T_years * 365))
    return FxVanillaOption(
        pair="EURUSD",
        strike=K,
        expiry=expiry,
        option_type=option_type,
        notional=notional,
        trade_date=TRADE_DATE,
    )


def _snapshot(S, r_d, r_f, vol):
    return FakeSnapshot(date=TRADE_DATE, spot=S, r_d=r_d, r_f=r_f, flat_vol=vol)


@pytest.mark.parametrize(
    # T values are exact multiples of 1/365 so FxVanillaOption's integer-day-count
    # time_to_expiry introduces no rounding drift versus the hand-computed reference.
    "S,K,r_d,r_f,vol,T,expected_call,expected_put",
    [
        (42, 40, 0.10, 0.0, 0.20, 182 / 365, 4.753175, 0.807565),
        (1.10, 1.10, 0.03, 0.01, 0.10, 1.0, 0.054638, 0.033073),
    ],
)
def test_reference_values(S, K, r_d, r_f, vol, T, expected_call, expected_put):
    snapshot = _snapshot(S, r_d, r_f, vol)
    call = _make_option(K, T, "call")
    put = _make_option(K, T, "put")

    assert PRICER.price(call, snapshot) == pytest.approx(expected_call, abs=1e-4)
    assert PRICER.price(put, snapshot) == pytest.approx(expected_put, abs=1e-4)


@pytest.mark.parametrize(
    "S,K,r_d,r_f,vol,T",
    [
        (42, 40, 0.10, 0.0, 0.20, 0.5),
        (1.10, 1.10, 0.03, 0.01, 0.10, 1.0),
        (1.25, 1.30, 0.02, 0.045, 0.15, 0.25),
    ],
)
def test_put_call_parity(S, K, r_d, r_f, vol, T):
    snapshot = _snapshot(S, r_d, r_f, vol)
    call = _make_option(K, T, "call")
    put = _make_option(K, T, "put")
    # Use the option's actual (integer-day-rounded) time-to-expiry, not the nominal
    # T passed in, since FxVanillaOption rounds to whole calendar days.
    actual_T = call.time_to_expiry(TRADE_DATE)

    lhs = PRICER.price(call, snapshot) - PRICER.price(put, snapshot)
    rhs = S * math.exp(-r_f * actual_T) - K * math.exp(-r_d * actual_T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_via_finite_difference(option_type):
    S, K, r_d, r_f, vol, T = 1.20, 1.18, 0.025, 0.01, 0.12, 0.75
    option = _make_option(K, T, option_type)
    snapshot = _snapshot(S, r_d, r_f, vol)
    greeks = PRICER.greeks(option, snapshot)

    eps_s = S * 1e-5
    p_up = PRICER.price(option, _snapshot(S + eps_s, r_d, r_f, vol))
    p_dn = PRICER.price(option, _snapshot(S - eps_s, r_d, r_f, vol))
    fd_delta = (p_up - p_dn) / (2 * eps_s)
    fd_gamma = (p_up - 2 * PRICER.price(option, snapshot) + p_dn) / (eps_s**2)
    assert greeks.delta == pytest.approx(fd_delta, abs=1e-4)
    assert greeks.gamma == pytest.approx(fd_gamma, abs=1e-2)

    eps_v = 1e-5
    v_up = PRICER.price(option, _snapshot(S, r_d, r_f, vol + eps_v))
    v_dn = PRICER.price(option, _snapshot(S, r_d, r_f, vol - eps_v))
    fd_vega = (v_up - v_dn) / (2 * eps_v)
    assert greeks.vega == pytest.approx(fd_vega, abs=1e-3)

    # Theta: shorten expiry by exactly 1 day (finest resolution the day-count
    # granularity of FxVanillaOption.time_to_expiry supports) and re-price.
    expiry_days = round(T * 365)
    option_shorter = FxVanillaOption(
        pair="EURUSD", strike=K, expiry=TRADE_DATE + pd.Timedelta(days=expiry_days - 1),
        option_type=option_type, notional=option.notional, trade_date=TRADE_DATE,
    )
    # theta = dV/d(calendar time) = -dV/dT ~= (V(T-dt) - V(T)) / dt
    dt = 1 / 365.0
    p_base = PRICER.price(option, snapshot)
    p_shorter = PRICER.price(option_shorter, snapshot)
    fd_theta = (p_shorter - p_base) / dt
    assert greeks.theta == pytest.approx(fd_theta, abs=5e-2)


def test_expired_option_prices_to_intrinsic():
    S, K = 1.20, 1.15
    snapshot = _snapshot(S, 0.02, 0.01, 0.10)
    expiry = TRADE_DATE
    call = FxVanillaOption(
        pair="EURUSD", strike=K, expiry=expiry, option_type="call",
        notional=1.0, trade_date=TRADE_DATE,
    )
    assert PRICER.price(call, snapshot) == pytest.approx(max(S - K, 0.0))
    greeks = PRICER.greeks(call, snapshot)
    assert (greeks.delta, greeks.gamma, greeks.vega, greeks.theta) == (0.0, 0.0, 0.0, 0.0)
