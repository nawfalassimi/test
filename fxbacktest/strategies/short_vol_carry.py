from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from fxbacktest.execution.order import Order
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.strategies.base import Strategy, register_strategy

if TYPE_CHECKING:
    from fxbacktest.market.snapshot import MarketSnapshot
    from fxbacktest.portfolio.portfolio import Portfolio


@register_strategy("short_vol_carry_1m")
class ShortVolCarryStrategy(Strategy):
    """Sell a 1-month ATM-forward straddle every Monday, held to expiry.
    Skips the week's entry if a previous straddle from this strategy is still
    open (no overlapping clips in v1 — a documented simplification)."""

    def __init__(self, pair: str = "EURUSD", tenor_days: int = 30, entry_weekday: int = 0,
                 notional: float = 1_000_000, strategy_id: str = "short_vol_carry_1m"):
        self.pair = pair
        self.tenor_days = tenor_days
        self.entry_weekday = entry_weekday
        self.notional = notional
        self.strategy_id = strategy_id

    def generate_orders(self, date: pd.Timestamp, snapshot: "MarketSnapshot",
                        portfolio: "Portfolio") -> List[Order]:
        if date.weekday() != self.entry_weekday:
            return []
        if portfolio.has_open_position(self.strategy_id):
            return []

        T = self.tenor_days / 365.0
        strike = snapshot.forward(T)
        expiry = date + pd.Timedelta(days=self.tenor_days)
        clip_id = f"{self.strategy_id}_{date:%Y%m%d}"

        call = FxVanillaOption(pair=self.pair, strike=strike, expiry=expiry, option_type="call",
                                notional=self.notional, trade_date=date)
        put = FxVanillaOption(pair=self.pair, strike=strike, expiry=expiry, option_type="put",
                              notional=self.notional, trade_date=date)

        return [
            Order(instrument=call, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id),
            Order(instrument=put, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id),
        ]
