from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from fxbacktest.data.schema import TENOR_YEARS
from fxbacktest.market.curve import FxForwardCurve
from fxbacktest.market.vol_surface import VolSurface

# Single (r_d, r_f) pair per date: r_f is fixed by config (assumed_foreign_rate)
# and r_d is derived once from the forward curve's implied carry at a single
# reference tenor ("1M", the tenor this milestone's only strategy actually
# trades). Exact for 1M instruments; a documented approximation for other
# tenors, revisited if/when multi-tenor strategies are added.
REFERENCE_TENOR = "1M"


@dataclass(frozen=True)
class MarketSnapshot:
    date: pd.Timestamp
    pair: str
    spot: float
    forward_curve: FxForwardCurve
    vol_surface: VolSurface
    r_d: float
    r_f: float

    def forward(self, T: float) -> float:
        return self.forward_curve.forward(T)

    def discount_factor(self, T: float, domestic: bool = True) -> float:
        r = self.r_d if domestic else self.r_f
        return math.exp(-r * T)

    def implied_vol_for_strike(self, K: float, T: float) -> float:
        return self.vol_surface.get_vol_for_strike(K, self.spot, self.r_d, self.r_f, T)

    def vol_for_delta(self, delta_call: float, T: float) -> float:
        return self.vol_surface.get_vol(delta_call, T)


def build_market_snapshot(date: pd.Timestamp, quotes_df: pd.DataFrame, pair: str,
                          assumed_foreign_rate: float = 0.0) -> MarketSnapshot:
    """Pure function: the single construction point for MarketSnapshot, so every
    layer of the pipeline sees the same immutable view of the world for a date.

    quotes_df may contain quotes for multiple currency pairs (e.g. when
    trading several pairs in one backtest) — always filter by BOTH date and
    pair, never date alone, or rows from different pairs get silently mixed
    together into one (nonsensical) curve/vol surface."""
    day_quotes = quotes_df[(quotes_df["date"] == date) & (quotes_df["pair"] == pair)]
    if day_quotes.empty:
        raise ValueError(f"no quotes found for pair {pair} on date {date}")

    spot = float(day_quotes["spot"].iloc[0])

    tenor_fwd_points = day_quotes.set_index("tenor")["fwd_points"].to_dict()
    curve = FxForwardCurve(spot=spot, tenor_fwd_points=tenor_fwd_points)

    tenor_quotes = {
        row.tenor: {"atm_vol": row.atm_vol, "rr25": row.rr25, "bf25": row.bf25,
                    "rr10": row.rr10, "bf10": row.bf10}
        for row in day_quotes.itertuples()
    }
    surface = VolSurface(tenor_quotes)

    r_f = assumed_foreign_rate
    r_d = r_f + curve.implied_carry(TENOR_YEARS[REFERENCE_TENOR])

    return MarketSnapshot(date=date, pair=pair, spot=spot, forward_curve=curve,
                           vol_surface=surface, r_d=r_d, r_f=r_f)
