from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from fxbacktest.execution.order import Order
from fxbacktest.instruments.spot import FxSpot

if TYPE_CHECKING:
    from fxbacktest.market.snapshot import MarketSnapshot
    from fxbacktest.portfolio.portfolio import Portfolio
    from fxbacktest.pricing.base import Pricer


class DailyDeltaHedger:
    """Standing hedging process, independent of any strategy. In "daily" mode,
    flattens net portfolio delta to (near) zero every trading day. In
    "threshold" mode, only trades when |net_delta| exceeds a configured limit."""

    def __init__(self, pricer: "Pricer", mode: str = "daily", threshold: float = 0.0,
                 strategy_id: str = "hedge"):
        self.pricer = pricer
        self.mode = mode
        self.threshold = threshold
        self.strategy_id = strategy_id

    def rehedge_orders(self, date: pd.Timestamp, snapshot: "MarketSnapshot",
                        portfolio: "Portfolio") -> List[Order]:
        net_delta = portfolio.net_delta(snapshot, self.pricer)

        if self.mode == "daily":
            should_hedge = abs(net_delta) > 1e-9
        elif self.mode == "threshold":
            should_hedge = abs(net_delta) > self.threshold
        else:
            raise ValueError(f"unknown hedging mode: {self.mode}")

        if not should_hedge:
            return []

        # net_delta > 0 => net long base currency exposure => sell spot to flatten.
        side = "sell" if net_delta > 0 else "buy"
        instrument = FxSpot(pair=snapshot.pair, notional=abs(net_delta))
        clip_id = f"hedge_{date:%Y%m%d}"
        return [Order(instrument=instrument, side=side, qty=1.0, clip_id=clip_id,
                      strategy_id=self.strategy_id, order_type="hedge")]
