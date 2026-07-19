from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from fxbacktest.data.schema import parse_pair
from fxbacktest.execution.order import Order
from fxbacktest.instruments.spot import FxSpot

if TYPE_CHECKING:
    from fxbacktest.market.market import Market
    from fxbacktest.portfolio.portfolio import Portfolio
    from fxbacktest.pricing.base import Pricer


class DailyDeltaHedger:
    """Standing hedging process, independent of any strategy. In "daily" mode,
    flattens net portfolio delta to (near) zero every trading day, per pair.
    In "threshold" mode, only trades a pair whose USD-equivalent delta exceeds
    a configured limit. Hedges are always sized in each pair's own native
    (base-currency) notional — you cannot flatten EUR risk with a USDJPY spot
    trade, so hedging must stay pair-by-pair even though risk is reported in
    USD (see Portfolio.native_delta_by_pair vs Portfolio.net_delta)."""

    def __init__(self, pricer: "Pricer", mode: str = "daily", threshold: float = 0.0,
                 strategy_id: str = "hedge"):
        self.pricer = pricer
        self.mode = mode
        self.threshold = threshold
        self.strategy_id = strategy_id

    def rehedge_orders(self, date: pd.Timestamp, market: "Market",
                        portfolio: "Portfolio") -> List[Order]:
        native_deltas = portfolio.native_delta_by_pair(market, self.pricer)

        orders = []
        for pair, native_delta in native_deltas.items():
            if abs(native_delta) < 1e-9:
                continue

            if self.mode == "daily":
                should_hedge = True
            elif self.mode == "threshold":
                base, _ = parse_pair(pair)
                usd_delta = native_delta * market.usd_rate(base)
                should_hedge = abs(usd_delta) > self.threshold
            else:
                raise ValueError(f"unknown hedging mode: {self.mode}")

            if not should_hedge:
                continue

            # native_delta > 0 => net long base currency exposure => sell spot to flatten.
            side = "sell" if native_delta > 0 else "buy"
            instrument = FxSpot(pair=pair, notional=abs(native_delta))
            clip_id = f"hedge_{pair}_{date:%Y%m%d}"
            orders.append(Order(instrument=instrument, side=side, qty=1.0, clip_id=clip_id,
                                strategy_id=self.strategy_id, order_type="hedge"))
        return orders
