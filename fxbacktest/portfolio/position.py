from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from fxbacktest.instruments.base import Instrument


@dataclass
class Position:
    instrument: Instrument
    qty: float  # signed: positive = long, negative = short
    clip_id: str
    strategy_id: str
    entry_date: pd.Timestamp
    entry_price: float  # actual executed price (post transaction-cost adjustment)
    entry_vol: Optional[float] = None
    last_mark_price: Optional[float] = None  # last price used to accrue P&L; defaults to entry_price
    cost_paid: float = 0.0  # dollar transaction cost at execution, always >= 0; set once, never re-derived
    is_open: bool = True
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    def __post_init__(self) -> None:
        if self.last_mark_price is None:
            self.last_mark_price = self.entry_price
