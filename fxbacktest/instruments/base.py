from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument(ABC):
    """Dumb data container. Pricing logic lives in fxbacktest.pricing, never here."""

    pair: str
