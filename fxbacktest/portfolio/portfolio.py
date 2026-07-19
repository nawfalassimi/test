from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from fxbacktest.data.schema import parse_pair
from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.pricing.base import Greeks
from fxbacktest.portfolio.position import Position

if TYPE_CHECKING:
    from fxbacktest.execution.order import Order
    from fxbacktest.market.market import Market
    from fxbacktest.pricing.base import Pricer

_HEDGE_FLAT_EPS = 1e-9


@dataclass
class Portfolio:
    """No cash account: cum_pnl starts at 0 and simply accrues +/- as
    positions are marked to market, via each position's own last_mark_price.

    All money amounts (prices, P&L, cost_paid) and all greeks are expressed
    in USD, regardless of the instrument's own currency pair — see
    instrument_value/instrument_greeks. This is what lets positions in
    different pairs be summed together directly throughout this class."""

    positions: List[Position] = field(default_factory=list)
    cum_pnl: float = 0.0

    def instrument_value(self, instrument, market: "Market", pricer: "Pricer") -> float:
        """Fair value in USD. Price/premium is natively in the pair's QUOTE
        currency (confirmed in GarmanKohlhagenPricer), so convert via the
        quote currency's USD rate."""
        snapshot = market.snapshot(instrument.pair)
        _, quote = parse_pair(instrument.pair)
        native = instrument.notional * snapshot.spot if isinstance(instrument, FxSpot) else pricer.price(instrument, snapshot)
        return native * market.usd_rate(quote)

    def instrument_greeks(self, instrument, market: "Market", pricer: "Pricer") -> Greeks:
        """Greeks in USD. Delta/gamma are natively in the pair's BASE currency
        (an FxSpot's delta is literally its base-currency notional); vega/theta
        are natively in the QUOTE currency (same units as price, since they
        are dPrice/dVol and dPrice/dt)."""
        snapshot = market.snapshot(instrument.pair)
        base, quote = parse_pair(instrument.pair)
        base_rate = market.usd_rate(base)
        quote_rate = market.usd_rate(quote)
        if isinstance(instrument, FxSpot):
            native = Greeks(delta=instrument.notional, gamma=0.0, vega=0.0, theta=0.0)
        else:
            native = pricer.greeks(instrument, snapshot)
        return Greeks(delta=native.delta * base_rate, gamma=native.gamma * base_rate,
                      vega=native.vega * quote_rate, theta=native.theta * quote_rate)

    def native_delta_by_pair(self, market: "Market", pricer: "Pricer") -> Dict[str, float]:
        """Each open position's delta in its OWN pair's native base currency,
        grouped by pair — deliberately bypassing instrument_greeks' USD
        conversion. Used for hedging: you cannot flatten EUR risk with a
        USDJPY spot trade, so hedge sizing must stay in native currency."""
        totals: Dict[str, float] = defaultdict(float)
        for pos in self.positions:
            if not pos.is_open:
                continue
            pair = pos.instrument.pair
            if isinstance(pos.instrument, FxSpot):
                native_delta = pos.instrument.notional
            else:
                snapshot = market.snapshot(pair)
                native_delta = pricer.greeks(pos.instrument, snapshot).delta
            totals[pair] += pos.qty * native_delta
        return dict(totals)

    def mark_to_market(self, market: "Market", pricer: "Pricer") -> Dict[str, float]:
        """Accrue each open position's P&L since it was last marked (or since
        entry, for a position opened earlier today), update its last_mark_price,
        settle any option that has matured as of this date, and consolidate any
        open hedge positions into a single net position per pair. Call once per
        day, after that day's trades, so this reflects the end-of-day book."""
        daily_pnl = total_delta = total_gamma = total_vega = total_theta = 0.0
        friction_cost = 0.0
        for pos in self.positions:
            if pos.entry_date == market.date:
                friction_cost += pos.cost_paid
            if not pos.is_open:
                continue
            value = self.instrument_value(pos.instrument, market, pricer)
            greeks = self.instrument_greeks(pos.instrument, market, pricer)

            daily_pnl += pos.qty * (value - pos.last_mark_price)
            pos.last_mark_price = value

            total_delta += pos.qty * greeks.delta
            total_gamma += pos.qty * greeks.gamma
            total_vega += pos.qty * greeks.vega
            total_theta += pos.qty * greeks.theta

            if isinstance(pos.instrument, FxVanillaOption) and pos.instrument.is_expired(market.date):
                pos.is_open = False
                pos.exit_date = market.date
                pos.exit_price = value
                pos.realized_pnl = pos.qty * (value - pos.entry_price)

        self.cum_pnl += daily_pnl
        self._consolidate_hedge_positions(market, pricer)
        return {
            "date": market.date, "pnl": daily_pnl, "cum_pnl": self.cum_pnl,
            "delta": total_delta, "gamma": total_gamma, "vega": total_vega, "theta": total_theta,
            "fx_exposure": total_delta, "friction_cost": friction_cost,
        }

    def _consolidate_hedge_positions(self, market: "Market", pricer: "Pricer",
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

            combined_exposure = sum(p.qty * p.instrument.notional for p in positions)  # native base ccy
            strategy_id = positions[0].strategy_id
            for p in positions:
                value = self.instrument_value(p.instrument, market, pricer)
                p.is_open = False
                p.exit_date = market.date
                p.exit_price = value
                p.realized_pnl = p.qty * (value - p.entry_price)

            if abs(combined_exposure) < _HEDGE_FLAT_EPS:
                continue

            new_qty = 1.0 if combined_exposure > 0 else -1.0
            new_instrument = FxSpot(pair=pair, notional=abs(combined_exposure))
            fair_value = self.instrument_value(new_instrument, market, pricer)
            self.positions.append(Position(
                instrument=new_instrument, qty=new_qty,
                clip_id=f"hedge_consolidated_{pair}_{market.date:%Y%m%d}",
                strategy_id=strategy_id, entry_date=market.date,
                entry_price=fair_value, cost_paid=0.0,
            ))

    def net_delta(self, market: "Market", pricer: "Pricer") -> float:
        """Aggregate USD delta across every open position, in every pair —
        the "all risk in USD" reporting number."""
        return sum(
            pos.qty * self.instrument_greeks(pos.instrument, market, pricer).delta
            for pos in self.positions if pos.is_open
        )

    def has_open_position(self, strategy_id: str, *, pair: Optional[str] = None) -> bool:
        return any(pos.is_open and pos.strategy_id == strategy_id
                  and (pair is None or pos.instrument.pair == pair) for pos in self.positions)

    def execute(self, orders: List["Order"], market: "Market", pricer: "Pricer",
                cost_model: Optional[TransactionCostModel] = None) -> None:
        cost_model = cost_model or TransactionCostModel.zero()

        for order in orders:
            signed_qty = order.signed_qty
            fair_price = self.instrument_value(order.instrument, market, pricer)  # USD
            pair = order.instrument.pair
            snapshot = market.snapshot(pair)

            _, quote = parse_pair(pair)
            quote_rate = market.usd_rate(quote)

            if isinstance(order.instrument, FxVanillaOption):
                vega = self.instrument_greeks(order.instrument, market, pricer).vega  # USD
                adjustment = cost_model.option_cost(pair, order.side, fair_price, vega)
            else:
                # spot_cost's formula (half-spread-in-pips * base-currency notional)
                # naturally yields a QUOTE-currency amount (a pip is a quote/base
                # price increment) — pass the native base notional in, matching
                # option_cost's contract, then convert the native QUOTE-currency
                # result to USD the same way fair_price already is. Converting the
                # notional itself to USD first (rather than the resulting cost)
                # would silently change the formula's meaning.
                adjustment = cost_model.spot_cost(pair, order.side, order.instrument.notional) * quote_rate

            entry_price = fair_price + adjustment  # USD
            cost_paid = abs(adjustment)  # USD
            entry_vol = None
            if isinstance(order.instrument, FxVanillaOption):
                T = order.instrument.time_to_expiry(market.date)
                if T > 0:
                    entry_vol = snapshot.implied_vol_for_strike(order.instrument.strike, T)

            position = Position(
                instrument=order.instrument, qty=signed_qty, clip_id=order.clip_id,
                strategy_id=order.strategy_id, entry_date=market.date,
                entry_price=entry_price, entry_vol=entry_vol, cost_paid=cost_paid,
            )
            self.positions.append(position)
