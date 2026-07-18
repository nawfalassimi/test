from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.pricing.base import Greeks, Pricer
from fxbacktest.pricing.black_scholes_utils import d1 as _d1
from fxbacktest.pricing.black_scholes_utils import d2 as _d2
from fxbacktest.pricing.black_scholes_utils import norm_cdf, norm_pdf

if TYPE_CHECKING:
    from fxbacktest.market.snapshot import MarketSnapshot

# Uses spot (unadjusted) delta convention, not premium-adjusted delta.
# Standard FX-market simplification for v1; premium-adjusted delta is a v2 TODO.
#
# Theta sign convention: theta = dV/d(calendar time), i.e. the same signed
# derivative finite-differenced in test_gk_pricer.py — NOT the "cost of decay"
# convention (-dV/dt) some desks quote. A held-to-expiry short option has
# theta > 0 here (value falls as time passes while short). The classic
# delta-hedged P&L identity in this convention is therefore
# pnl ~= 0.5*gamma*(dS)^2 + theta*dt (theta ADDED, not subtracted) — see
# test_engine_integration.py.


class GarmanKohlhagenPricer(Pricer):
    def _inputs(self, instrument: FxVanillaOption, snapshot: "MarketSnapshot"):
        T = instrument.time_to_expiry(snapshot.date)
        S = snapshot.spot
        K = instrument.strike
        r_d, r_f = snapshot.r_d, snapshot.r_f
        vol = snapshot.implied_vol_for_strike(K, T) if T > 0 else 0.0
        return S, K, r_d, r_f, vol, T

    def price(self, instrument: FxVanillaOption, snapshot: "MarketSnapshot") -> float:
        if not isinstance(instrument, FxVanillaOption):
            raise TypeError(f"GarmanKohlhagenPricer cannot price {type(instrument)}")

        S, K, r_d, r_f, vol, T = self._inputs(instrument, snapshot)
        if T <= 0:
            return instrument.intrinsic_value(S) * instrument.notional

        dd1, dd2 = _d1(S, K, r_d, r_f, vol, T), _d2(S, K, r_d, r_f, vol, T)
        if instrument.option_type == "call":
            price = S * math.exp(-r_f * T) * norm_cdf(dd1) - K * math.exp(-r_d * T) * norm_cdf(dd2)
        else:
            price = K * math.exp(-r_d * T) * norm_cdf(-dd2) - S * math.exp(-r_f * T) * norm_cdf(-dd1)
        return price * instrument.notional

    def greeks(self, instrument: FxVanillaOption, snapshot: "MarketSnapshot") -> Greeks:
        if not isinstance(instrument, FxVanillaOption):
            raise TypeError(f"GarmanKohlhagenPricer cannot price {type(instrument)}")

        S, K, r_d, r_f, vol, T = self._inputs(instrument, snapshot)
        if T <= 0:
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)

        dd1, dd2 = _d1(S, K, r_d, r_f, vol, T), _d2(S, K, r_d, r_f, vol, T)
        disc_f = math.exp(-r_f * T)
        disc_d = math.exp(-r_d * T)
        pdf_d1 = norm_pdf(dd1)

        gamma = disc_f * pdf_d1 / (S * vol * math.sqrt(T))
        vega = S * disc_f * pdf_d1 * math.sqrt(T)

        if instrument.option_type == "call":
            delta = disc_f * norm_cdf(dd1)
            theta = (
                -S * disc_f * pdf_d1 * vol / (2 * math.sqrt(T))
                - r_d * K * disc_d * norm_cdf(dd2)
                + r_f * S * disc_f * norm_cdf(dd1)
            )
        else:
            delta = disc_f * (norm_cdf(dd1) - 1)
            theta = (
                -S * disc_f * pdf_d1 * vol / (2 * math.sqrt(T))
                + r_d * K * disc_d * norm_cdf(-dd2)
                - r_f * S * disc_f * norm_cdf(-dd1)
            )

        n = instrument.notional
        return Greeks(delta=delta * n, gamma=gamma * n, vega=vega * n, theta=theta * n)
