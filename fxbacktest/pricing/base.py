from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fxbacktest.instruments.base import Instrument
    from fxbacktest.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


class Pricer(ABC):
    @abstractmethod
    def price(self, instrument: "Instrument", snapshot: "MarketSnapshot") -> float:
        ...

    @abstractmethod
    def greeks(self, instrument: "Instrument", snapshot: "MarketSnapshot") -> Greeks:
        ...
