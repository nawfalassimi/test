from __future__ import annotations

import numpy as np
import pandas as pd

from fxbacktest.data.schema import QUOTE_COLUMNS, TENOR_YEARS

# Per-tenor multipliers giving a mildly upward-sloping term structure for ATM vol
# and a skew/convexity term structure that flattens at longer tenors. Toy values,
# not calibrated to any real market.
_ATM_TENOR_MULT = {"1W": 0.95, "1M": 1.00, "3M": 1.03, "6M": 1.06, "1Y": 1.10}
_SKEW_TENOR_MULT = {"1W": 1.10, "1M": 1.00, "3M": 0.90, "6M": 0.80, "1Y": 0.65}


def _ou_process(rng: np.random.Generator, n: int, x0: float, mean: float,
                 kappa: float, sigma: float, floor: float = None, cap: float = None) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] + kappa * (mean - x[i - 1]) + sigma * rng.standard_normal()
        if floor is not None:
            x[i] = max(x[i], floor)
        if cap is not None:
            x[i] = min(x[i], cap)
    return x


class SyntheticFxDataGenerator:
    """Generates a toy but internally-consistent close-of-day FX quote history
    (spot, forward points, ATM/RR/BF vol quotes) for one currency pair, so the
    backtest pipeline can be built and tested without a real data feed."""

    def __init__(
        self,
        pair: str = "EURUSD",
        start: str = "2022-01-01",
        end: str = "2023-12-29",
        seed: int = 42,
        base_spot: float = 1.10,
        spot_daily_vol: float = 0.005,
        base_carry: float = 0.015,
        base_atm_vol: float = 0.075,
        atm_vol_floor: float = 0.04,
    ):
        self.pair = pair
        self.dates = pd.bdate_range(start, end)
        self.seed = seed
        self.base_spot = base_spot
        self.spot_daily_vol = spot_daily_vol
        self.base_carry = base_carry
        self.base_atm_vol = base_atm_vol
        self.atm_vol_floor = atm_vol_floor
        self.rng = np.random.default_rng(seed)

    def generate_spot_path(self) -> pd.Series:
        n = len(self.dates)
        daily_returns = self.rng.normal(loc=0.0, scale=self.spot_daily_vol, size=n)
        daily_returns[0] = 0.0
        spot = self.base_spot * np.exp(np.cumsum(daily_returns))
        return pd.Series(spot, index=self.dates, name="spot")

    def generate_carry_path(self) -> pd.Series:
        n = len(self.dates)
        carry = _ou_process(self.rng, n, x0=self.base_carry, mean=self.base_carry,
                             kappa=0.02, sigma=0.0008)
        return pd.Series(carry, index=self.dates, name="carry")

    def generate_forward_points(self, spot: pd.Series, carry: pd.Series) -> pd.DataFrame:
        rows = []
        for tenor, T in TENOR_YEARS.items():
            fwd = spot.values * np.exp(carry.values * T)
            fwd_points = (fwd - spot.values) * 10_000.0
            rows.append(pd.DataFrame({"date": self.dates, "tenor": tenor, "fwd_points": fwd_points}))
        return pd.concat(rows, ignore_index=True)

    def generate_vol_surface_quotes(self) -> pd.DataFrame:
        n = len(self.dates)
        atm_level = _ou_process(self.rng, n, x0=self.base_atm_vol, mean=self.base_atm_vol,
                                 kappa=0.05, sigma=0.0025, floor=self.atm_vol_floor)
        rr25_level = _ou_process(self.rng, n, x0=0.0, mean=0.0,
                                  kappa=0.03, sigma=0.0006, floor=-0.01, cap=0.01)
        bf25_level = _ou_process(self.rng, n, x0=0.0025, mean=0.0025,
                                  kappa=0.03, sigma=0.0003, floor=0.0005, cap=0.01)

        rows = []
        for tenor in TENOR_YEARS:
            noise = lambda scale: self.rng.normal(0.0, scale, size=n)
            atm_vol = atm_level * _ATM_TENOR_MULT[tenor] + noise(0.001)
            atm_vol = np.maximum(atm_vol, self.atm_vol_floor)

            rr25 = rr25_level * _SKEW_TENOR_MULT[tenor] + noise(0.0003)
            bf25 = np.maximum(bf25_level * _SKEW_TENOR_MULT[tenor] + noise(0.0002), 0.0002)
            # rr10 magnitude larger than rr25 (10-delta wings more skewed).
            rr10 = rr25 * 1.4 + noise(0.0002)
            # bf10 >= bf25 enforced by construction: bf25 plus a strictly positive extra term.
            bf10_extra = np.abs(noise(0.0006)) + 0.0008
            bf10 = bf25 + bf10_extra

            rows.append(pd.DataFrame({
                "date": self.dates, "tenor": tenor,
                "atm_vol": atm_vol, "rr25": rr25, "bf25": bf25, "rr10": rr10, "bf10": bf10,
            }))
        return pd.concat(rows, ignore_index=True)

    def generate(self) -> pd.DataFrame:
        spot = self.generate_spot_path()
        carry = self.generate_carry_path()
        fwd_df = self.generate_forward_points(spot, carry)
        vol_df = self.generate_vol_surface_quotes()

        df = fwd_df.merge(vol_df, on=["date", "tenor"])
        df["spot"] = df["date"].map(spot)
        df["pair"] = self.pair
        return df[["pair"] + QUOTE_COLUMNS].sort_values(["date", "tenor"]).reset_index(drop=True)
