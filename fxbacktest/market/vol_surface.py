from __future__ import annotations

import math
from typing import Dict

import numpy as np

from fxbacktest.data.schema import TENOR_YEARS
from fxbacktest.market.smile import build_smile_points, interpolate_smile, solve_delta_for_strike


class VolSurface:
    """Vol surface for one currency pair on one date, built from per-tenor
    (atm_vol, rr25, bf25, rr10, bf10) quotes.

    get_vol(delta_call, T) interpolates across delta at each tenor pillar first
    (via each pillar's smile spline), then interpolates the resulting total
    variance (vol^2 * T) linearly across tenor to the target T, and converts
    back to vol. Interpolating in variance (not vol) avoids calendar arbitrage
    for expiries off the standard tenor grid, per the README's core gotcha.
    """

    def __init__(self, tenor_quotes: Dict[str, Dict[str, float]]):
        self._smile_fns = {}
        for tenor, q in tenor_quotes.items():
            points = build_smile_points(q["atm_vol"], q["rr25"], q["bf25"], q["rr10"], q["bf10"])
            self._smile_fns[tenor] = interpolate_smile(points)

        pairs = sorted((TENOR_YEARS[t], t) for t in tenor_quotes)
        self._pillar_T = np.array([p[0] for p in pairs])
        self._pillar_tenors = [p[1] for p in pairs]

    def get_vol(self, delta_call: float, T: float) -> float:
        pillar_vols = np.array([self._smile_fns[tenor](delta_call) for tenor in self._pillar_tenors])
        pillar_variance = pillar_vols**2 * self._pillar_T
        w_T = float(np.interp(T, self._pillar_T, pillar_variance))
        return math.sqrt(w_T / T)

    def get_vol_for_strike(self, K: float, S: float, r_d: float, r_f: float, T: float) -> float:
        """Implied vol for an arbitrary strike, via the delta<->strike fixed-point
        solver (needed since this surface is natively parametrized by delta)."""
        vol_fn = lambda delta_call: self.get_vol(delta_call, T)  # noqa: E731
        _, vol = solve_delta_for_strike(K, S, r_d, r_f, T, vol_fn)
        return vol
