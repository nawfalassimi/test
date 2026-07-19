from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from fxbacktest.data.schema import TENOR_DAYS, TENOR_YEARS
from fxbacktest.execution.order import Order
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.market.smile import solve_strike_for_delta
from fxbacktest.strategies.base import Strategy, register_strategy
from fxbacktest.strategies.signals import SignalBundle, compute_signal_bundle

if TYPE_CHECKING:
    from fxbacktest.market.market import Market
    from fxbacktest.portfolio.portfolio import Portfolio

_DEFAULT_PAIR = "EURUSD"


@register_strategy("short_vol_signal")
class ShortVolSignalStrategy(Strategy):
    """Sell a strangle (target_delta, e.g. 10d) at `tenor` (e.g. 3M) in each
    of required_pairs, gated by a composite of 5 signals (need >=2/5) plus a
    mandatory RR-skew condition, with two hard stops that both block new
    entries AND force an early exit of an open position. Held to natural
    expiry otherwise. No pricer dependency: strike solving uses the
    snapshot's vol_surface directly, and exits are lazy (pending_close),
    resolved by Portfolio.mark_to_market."""

    def __init__(self, quotes_df: pd.DataFrame, vix_df: pd.DataFrame,
                 pair: str = _DEFAULT_PAIR, pairs: Optional[List[str]] = None,
                 tenor: str = "3M", target_delta: float = 0.10,
                 notional: float = 1_000_000, strategy_id: str = "short_vol_signal"):
        if pairs is not None:
            if pair != _DEFAULT_PAIR:
                raise ValueError("pass either pair= or pairs=, not both")
            deduped = list(dict.fromkeys(pairs))
            if len(deduped) != len(pairs):
                raise ValueError(f"pairs contains duplicates: {pairs}")
            self.required_pairs = deduped
        else:
            self.required_pairs = [pair]

        if tenor not in TENOR_YEARS:
            raise ValueError(f"unknown tenor {tenor!r}, must be one of {list(TENOR_YEARS)}")

        self.tenor = tenor
        self.T = TENOR_YEARS[tenor]
        self.target_delta = target_delta
        self.notional = notional
        self.strategy_id = strategy_id

        vix_series = vix_df.set_index("date")["vix"]
        self._signals: Dict[str, SignalBundle] = {
            p: compute_signal_bundle(quotes_df, vix_series, p, tenor) for p in self.required_pairs
        }

    def generate_orders(self, date: pd.Timestamp, market: "Market",
                        portfolio: "Portfolio") -> List[Order]:
        orders: List[Order] = []
        for pair in self.required_pairs:
            bundle = self._signals[pair]
            is_open = portfolio.has_open_position(self.strategy_id, pair=pair)

            if is_open:
                if bool(bundle.exit_now.get(date, False)):
                    portfolio.mark_for_early_close(self.strategy_id, pair)
                continue

            if not bool(bundle.entry_ok.get(date, False)):
                continue

            snapshot = market.snapshot(pair)
            vol_fn = lambda d: snapshot.vol_surface.get_vol(d, self.T)  # noqa: B023
            k_call, _ = solve_strike_for_delta(self.target_delta, snapshot.spot, snapshot.r_d,
                                                snapshot.r_f, self.T, vol_fn)
            k_put, _ = solve_strike_for_delta(1 - self.target_delta, snapshot.spot, snapshot.r_d,
                                               snapshot.r_f, self.T, vol_fn)
            expiry = date + pd.Timedelta(days=TENOR_DAYS[self.tenor])
            clip_id = f"{self.strategy_id}_{pair}_{date:%Y%m%d}"

            call = FxVanillaOption(pair=pair, strike=k_call, expiry=expiry, option_type="call",
                                   notional=self.notional, trade_date=date)
            put = FxVanillaOption(pair=pair, strike=k_put, expiry=expiry, option_type="put",
                                  notional=self.notional, trade_date=date)
            orders.append(Order(instrument=call, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id))
            orders.append(Order(instrument=put, side="sell", qty=1.0, clip_id=clip_id, strategy_id=self.strategy_id))

        return orders
