from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.market.snapshot import build_market_snapshot

if TYPE_CHECKING:
    from fxbacktest.portfolio.portfolio import Portfolio
    from fxbacktest.portfolio.position import Position
    from fxbacktest.pricing.base import Pricer

BLOTTER_COLUMNS = [
    "trade_id", "clip_id", "strategy_id", "pair", "instrument_type", "option_type",
    "entry_date", "date", "status", "entry_price", "current_price",
    "entry_vol", "current_vol", "delta", "vega", "gamma",
]


def _trade_id(position: "Position") -> str:
    if isinstance(position.instrument, FxVanillaOption):
        return f"{position.clip_id}_{position.instrument.option_type}"
    return f"{position.clip_id}_spot"


def _current_vol(position: "Position", snapshot) -> Optional[float]:
    if not isinstance(position.instrument, FxVanillaOption):
        return None
    T = position.instrument.time_to_expiry(snapshot.date)
    if T <= 0:
        return None
    return snapshot.implied_vol_for_strike(position.instrument.strike, T)


def _blotter_row(position: "Position", date: pd.Timestamp, snapshot, pricer: "Pricer",
                  portfolio: "Portfolio") -> dict:
    if position.entry_date == date:
        status = "new"
    elif not position.is_open and position.exit_date == date:
        status = "exit"
    else:
        status = "existing"

    current_price = portfolio.instrument_value(position.instrument, snapshot, pricer)
    greeks = portfolio.instrument_greeks(position.instrument, snapshot, pricer)

    return {
        "trade_id": _trade_id(position),
        "clip_id": position.clip_id,
        "strategy_id": position.strategy_id,
        "pair": position.instrument.pair,
        "instrument_type": type(position.instrument).__name__,
        "option_type": getattr(position.instrument, "option_type", None),
        "entry_date": position.entry_date,
        "date": date,
        "status": status,
        "entry_price": position.entry_price,
        "current_price": current_price,
        "entry_vol": position.entry_vol,
        "current_vol": _current_vol(position, snapshot),
        "delta": position.qty * greeks.delta,
        "vega": position.qty * greeks.vega,
        "gamma": position.qty * greeks.gamma,
    }


def build_trade_blotter(portfolio: "Portfolio", quotes_df: pd.DataFrame, pricer: "Pricer",
                         assumed_foreign_rate: float = 0.0,
                         exclude_strategy_ids: Iterable[str] = ()) -> pd.DataFrame:
    """One row per (date, position), for every date in [entry_date, exit_date
    (or the last backtest date, if still open)] that the position was open or
    closing. Built post-hoc (after the backtest completes) by re-pricing each
    active position once per date — cheap at milestone-1 data volumes, and
    keeps the daily loop itself untouched.

    Includes the delta hedger's clips (strategy_id="hedge") by default:
    Portfolio._consolidate_hedge_positions collapses all open hedge positions
    per pair down to at most one after every mark_to_market call, so a hedge
    position now lives for only a day or two before being replaced — the
    blotter's row count stays linear in the backtest length rather than the
    O(days^2) blow-up from when hedge positions accumulated forever. Pass
    exclude_strategy_ids=("hedge",) to hide hedge rows if only strategy trades
    are wanted (entry/current implied vol is always None for a spot position).
    """
    positions = [pos for pos in portfolio.positions if pos.strategy_id not in set(exclude_strategy_ids)]
    if not positions:
        return pd.DataFrame(columns=BLOTTER_COLUMNS)

    dates = pd.DatetimeIndex(sorted(quotes_df["date"].drop_duplicates().tolist()))
    last_date = dates[-1]

    rows = []
    for date in dates:
        active = [
            pos for pos in positions
            if pos.entry_date <= date <= (pos.exit_date if not pos.is_open else last_date)
        ]
        if not active:
            continue
        snapshot = build_market_snapshot(date, quotes_df, assumed_foreign_rate)
        rows.extend(_blotter_row(pos, date, snapshot, pricer, portfolio) for pos in active)

    return pd.DataFrame(rows, columns=BLOTTER_COLUMNS)
