from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from fxbacktest.execution.order import Order
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.strategies.base import Strategy, register_strategy

if TYPE_CHECKING:
    from fxbacktest.market.market import Market
    from fxbacktest.portfolio.portfolio import Portfolio

_DEFAULT_PAIR = "EURUSD"


@register_strategy("short_vol_carry_1m")
class ShortVolCarryStrategy(Strategy):
    """Sell a 1-month ATM-forward straddle every Monday, held to expiry, in
    each of required_pairs independently. Skips a given pair's entry for the
    week if a previous straddle from this strategy in THAT pair is still open
    (no overlapping clips per pair in v1 — a documented simplification; one
    pair being open never blocks another pair's entry)."""

    def __init__(self, pair: str = _DEFAULT_PAIR, pairs: Optional[List[str]] = None,
                 tenor_days: int = 30, entry_weekday: int = 0,
                 notional: float = 1_000_000, strategy_id: str = "short_vol_carry_1m"):
        if pairs is not None:
            if pair != _DEFAULT_PAIR:
                raise ValueError("pass either pair= or pairs=, not both")
            deduped = list(dict.fromkeys(pairs))
            if len(deduped) != len(pairs):
                raise ValueError(f"pairs contains duplicates: {pairs}")
            self.required_pairs = deduped
        else:
            self.required_pairs = [pair]

        self.tenor_days = tenor_days
        self.entry_weekday = entry_weekday
        self.notional = notional
        self.strategy_id = strategy_id

    def generate_orders(self, date: pd.Timestamp, market: "Market",
                        portfolio: "Portfolio") -> List[Order]:
        if date.weekday() != self.entry_weekday:
            return []

        orders: List[Order] = []
        for pair in self.required_pairs:
            if portfolio.has_open_position(self.strategy_id, pair=pair):
                continue

            snapshot = market.snapshot(pair)
            T = self.tenor_days / 365.0
            strike = snapshot.forward(T)
            expiry = date + pd.Timedelta(days=self.tenor_days)
            clip_id = f"{self.strategy_id}_{pair}_{date:%Y%m%d}"

            call = FxVanillaOption(pair=pair, strike=strike, expiry=expiry, option_type="call",
                                    notional=self.notional, trade_date=date)
            put = FxVanillaOption(pair=pair, strike=strike, expiry=expiry, option_type="put",
                                  notional=self.notional, trade_date=date)

            orders.append(Order(instrument=call, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id))
            orders.append(Order(instrument=put, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id))

        return orders
