from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from fxbacktest.instruments.base import Instrument

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class FxVanillaOption(Instrument):
    strike: float
    expiry: pd.Timestamp
    option_type: OptionType
    notional: float
    trade_date: pd.Timestamp

    def time_to_expiry(self, as_of: pd.Timestamp) -> float:
        days = (self.expiry - as_of).days
        return max(days, 0) / 365.0

    def is_expired(self, as_of: pd.Timestamp) -> bool:
        return as_of >= self.expiry

    def intrinsic_value(self, spot: float) -> float:
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)
