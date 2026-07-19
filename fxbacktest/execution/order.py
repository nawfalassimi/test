from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fxbacktest.instruments.base import Instrument

Side = Literal["buy", "sell"]
OrderType = Literal["open", "close", "hedge"]


@dataclass(frozen=True)
class Order:
    instrument: Instrument
    side: Side
    qty: float  # magnitude (always positive); direction comes from side
    clip_id: str
    strategy_id: str
    order_type: OrderType = "open"

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "buy" else -self.qty
