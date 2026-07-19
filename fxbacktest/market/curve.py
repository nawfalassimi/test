from __future__ import annotations

import math
from typing import Dict

import numpy as np

from fxbacktest.data.schema import TENOR_YEARS


class FxForwardCurve:
    """Forward curve for one currency pair on one date, built from spot and
    per-tenor forward points (in pips, i.e. forward = spot + fwd_points / 10_000).
    Interpolates forward points linearly across tenor pillars; T outside the
    pillar range is clipped to the nearest pillar (flat extrapolation) — a
    documented v1 simplification, fine given tenors span 1W-1Y."""

    def __init__(self, spot: float, tenor_fwd_points: Dict[str, float]):
        self.spot = spot
        self.tenor_fwd_points = tenor_fwd_points
        pairs = sorted((TENOR_YEARS[t], pts) for t, pts in tenor_fwd_points.items())
        self._pillar_T = np.array([p[0] for p in pairs])
        self._pillar_pts = np.array([p[1] for p in pairs])

    def _fwd_points(self, T: float) -> float:
        return float(np.interp(T, self._pillar_T, self._pillar_pts))

    def forward(self, T: float) -> float:
        return self.spot + self._fwd_points(T) / 10_000.0

    def implied_carry(self, T: float) -> float:
        """r_d - r_f, derived from F/S = exp((r_d - r_f) * T)."""
        F = self.forward(T)
        return math.log(F / self.spot) / T
