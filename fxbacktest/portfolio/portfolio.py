from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.pricing.base import Greeks
from fxbacktest.portfolio.position import Position

if TYPE_CHECKING:
    from fxbacktest.execution.order import Order
    from fxbacktest.market.snapshot import MarketSnapshot
    from fxbacktest.pricing.base import Pricer

_HEDGE_FLAT_EPS = 1e-9


@dataclass
class Portfolio:
    """No cash account: cum_pnl starts at 0 and simply accrues +/- as
    positions are marked to market, via each position's own last_mark_price."""

    positions: List[Position] = field(default_factory=list)
    cum_pnl: float = 0.0

    def instrument_value(self, instrument, snapshot: "MarketSnapshot", pricer: "Pricer") -> float:
        if isinstance(instrument, FxSpot):
            return instrument.notional * snapshot.spot
        return pricer.price(instrument, snapshot)

    def instrument_greeks(self, instrument, snapshot: "MarketSnapshot", pricer: "Pricer") -> Greeks:
        if isinstance(instrument, FxSpot):
            return Greeks(delta=instrument.notional, gamma=0.0, vega=0.0, theta=0.0)
        return pricer.greeks(instrument, snapshot)

    def mark_to_market(self, snapshot: "MarketSnapshot", pricer: "Pricer") -> Dict[str, float]:
        """Accrue each open position's P&L since it was last marked (or since
        entry, for a position opened earlier today), update its last_mark_price,
        settle any option that has matured as of this date, and consolidate any
        open hedge positions into a single net position per pair. Call once per
        day, after that day's trades, so this reflects the end-of-day book."""
        daily_pnl = total_delta = total_gamma = total_vega = total_theta = 0.0
        friction_cost = 0.0
        for pos in self.positions:
            if pos.entry_date == snapshot.date:
                friction_cost += pos.cost_paid
            if not pos.is_open:
                continue
            value = self.instrument_value(pos.instrument, snapshot, pricer)
            greeks = self.instrument_greeks(pos.instrument, snapshot, pricer)

            daily_pnl += pos.qty * (value - pos.last_mark_price)
            pos.last_mark_price = value

            total_delta += pos.qty * greeks.delta
            total_gamma += pos.qty * greeks.gamma
            total_vega += pos.qty * greeks.vega
            total_theta += pos.qty * greeks.theta

            if isinstance(pos.instrument, FxVanillaOption) and pos.instrument.is_expired(snapshot.date):
                pos.is_open = False
                pos.exit_date = snapshot.date
                pos.exit_price = value
                pos.realized_pnl = pos.qty * (value - pos.entry_price)

        self.cum_pnl += daily_pnl
        self._consolidate_hedge_positions(snapshot, pricer)
        return {
            "date": snapshot.date, "pnl": daily_pnl, "cum_pnl": self.cum_pnl,
            "delta": total_delta, "gamma": total_gamma, "vega": total_vega, "theta": total_theta,
            "fx_exposure": total_delta, "friction_cost": friction_cost,
        }

    def _consolidate_hedge_positions(self, snapshot: "MarketSnapshot", pricer: "Pricer",
                                      hedge_strategy_ids: Iterable[str] = ("hedge",)) -> None:
        """Collapse all open hedge-strategy FxSpot positions per pair into a
        single net position. Must run AFTER the accrual loop above (in
        mark_to_market), so every constituent's last_mark_price already equals
        its current fair value — closing them here is NAV-neutral bookkeeping
        (the same pattern as the expired-option settlement above it), not a
        market trade: the real transaction cost was already charged on the
        incremental hedge order that executed earlier the same day."""
        hedge_ids = set(hedge_strategy_ids)
        by_pair: Dict[str, List[Position]] = defaultdict(list)
        for pos in self.positions:
            if pos.is_open and pos.strategy_id in hedge_ids and isinstance(pos.instrument, FxSpot):
                by_pair[pos.instrument.pair].append(pos)

        for pair, positions in by_pair.items():
            if len(positions) <= 1:
                continue

            combined_exposure = sum(p.qty * p.instrument.notional for p in positions)
            strategy_id = positions[0].strategy_id
            for p in positions:
                value = self.instrument_value(p.instrument, snapshot, pricer)
                p.is_open = False
                p.exit_date = snapshot.date
                p.exit_price = value
                p.realized_pnl = p.qty * (value - p.entry_price)

            if abs(combined_exposure) < _HEDGE_FLAT_EPS:
                continue

            new_qty = 1.0 if combined_exposure > 0 else -1.0
            new_instrument = FxSpot(pair=pair, notional=abs(combined_exposure))
            fair_value = self.instrument_value(new_instrument, snapshot, pricer)
            self.positions.append(Position(
                instrument=new_instrument, qty=new_qty,
                clip_id=f"hedge_consolidated_{pair}_{snapshot.date:%Y%m%d}",
                strategy_id=strategy_id, entry_date=snapshot.date,
                entry_price=fair_value, cost_paid=0.0,
            ))

    def net_delta(self, snapshot: "MarketSnapshot", pricer: "Pricer") -> float:
        return sum(
            pos.qty * self.instrument_greeks(pos.instrument, snapshot, pricer).delta
            for pos in self.positions if pos.is_open
        )

    def has_open_position(self, strategy_id: str) -> bool:
        return any(pos.is_open and pos.strategy_id == strategy_id for pos in self.positions)

    def execute(self, orders: List["Order"], snapshot: "MarketSnapshot", pricer: "Pricer",
                cost_model: Optional[TransactionCostModel] = None) -> None:
        cost_model = cost_model or TransactionCostModel.zero()

        for order in orders:
            signed_qty = order.signed_qty
            fair_price = self.instrument_value(order.instrument, snapshot, pricer)
            pair = order.instrument.pair

            if isinstance(order.instrument, FxVanillaOption):
                vega = pricer.greeks(order.instrument, snapshot).vega
                adjustment = cost_model.option_cost(pair, order.side, fair_price, vega)
            else:
                adjustment = cost_model.spot_cost(pair, order.side, order.instrument.notional)

            entry_price = fair_price + adjustment
            cost_paid = abs(adjustment)
            entry_vol = None
            if isinstance(order.instrument, FxVanillaOption):
                T = order.instrument.time_to_expiry(snapshot.date)
                if T > 0:
                    entry_vol = snapshot.implied_vol_for_strike(order.instrument.strike, T)

            position = Position(
                instrument=order.instrument, qty=signed_qty, clip_id=order.clip_id,
                strategy_id=order.strategy_id, entry_date=snapshot.date,
                entry_price=entry_price, entry_vol=entry_vol, cost_paid=cost_paid,
            )
            self.positions.append(position)
