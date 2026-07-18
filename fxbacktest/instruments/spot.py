from __future__ import annotations

from dataclasses import dataclass

from fxbacktest.instruments.base import Instrument


@dataclass(frozen=True)
class FxSpot(Instrument):
    notional: float
