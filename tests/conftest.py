from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FakeSnapshot:
    """Minimal duck-typed stand-in for MarketSnapshot, for testing pricing in isolation
    before the real market layer (curve/vol_surface/snapshot) exists."""

    date: pd.Timestamp
    spot: float
    r_d: float
    r_f: float
    flat_vol: float

    def implied_vol_for_strike(self, K: float, T: float) -> float:
        return self.flat_vol
