from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from fxbacktest.analytics.metrics import compute_cum_pnl, compute_drawdown, max_drawdown

if TYPE_CHECKING:
    from fxbacktest.portfolio.portfolio import Portfolio

# Sharpe/Sortino/annualized-PnL below are computed directly on the daily
# dollar P&L series, not on a returns series — this system tracks no
# cash/AUM base (Portfolio has no cash account), so there is nothing to
# normalize P&L against. This is a common simplified convention for
# absolute-P&L overlay/carry strategies; treat these as scale-dependent
# (they'd change with notional) rather than true risk-adjusted returns.


def pnl_without_friction(result_df: pd.DataFrame) -> pd.Series:
    """Daily pnl already reflects the cost drag baked into entry_price at
    execution; adding back that day's friction_cost recovers the frictionless
    (gross) daily pnl."""
    return result_df["pnl"] + result_df["friction_cost"]


def cum_pnl_without_friction(result_df: pd.DataFrame) -> pd.Series:
    return pnl_without_friction(result_df).cumsum()


@dataclass
class DrawdownEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    max_drawdown: float  # <= 0
    n_days: int


def extract_drawdown_episodes(result_df: pd.DataFrame) -> List[DrawdownEpisode]:
    """Contiguous underwater periods. An episode's `end` is the date of full
    recovery back to the prior peak (not just the trough), so `n_days` is the
    length of the whole underwater period — matching a "Worst N Drawdowns"
    table with Start/End/Nb Days columns. An episode still underwater at the
    end of the backtest is included, using the last available date as `end`.
    """
    cum_pnl = compute_cum_pnl(result_df).reset_index(drop=True)
    dates = result_df["date"].reset_index(drop=True)
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max

    peak_indices = list(cum_pnl.index[drawdown == 0])
    n = len(cum_pnl)
    episodes: List[DrawdownEpisode] = []

    for i in range(len(peak_indices) - 1):
        start_idx, end_idx = peak_indices[i], peak_indices[i + 1]
        segment = drawdown.iloc[start_idx:end_idx + 1]
        if (segment < 0).any():
            episodes.append(DrawdownEpisode(
                start=dates.iloc[start_idx], end=dates.iloc[end_idx],
                max_drawdown=float(segment.min()),
                n_days=(dates.iloc[end_idx] - dates.iloc[start_idx]).days,
            ))

    if peak_indices and peak_indices[-1] != n - 1:
        start_idx = peak_indices[-1]
        segment = drawdown.iloc[start_idx:]
        if (segment < 0).any():
            episodes.append(DrawdownEpisode(
                start=dates.iloc[start_idx], end=dates.iloc[n - 1],
                max_drawdown=float(segment.min()),
                n_days=(dates.iloc[n - 1] - dates.iloc[start_idx]).days,
            ))

    return episodes


def average_drawdown(result_df: pd.DataFrame) -> float:
    episodes = extract_drawdown_episodes(result_df)
    if not episodes:
        return 0.0
    return float(np.mean([ep.max_drawdown for ep in episodes]))


def average_top_n_drawdown(result_df: pd.DataFrame, n: int = 5) -> float:
    episodes = sorted(extract_drawdown_episodes(result_df), key=lambda ep: ep.max_drawdown)
    top = episodes[:n]
    if not top:
        return 0.0
    return float(np.mean([ep.max_drawdown for ep in top]))


def drawdown_percentile(result_df: pd.DataFrame, pct: float = 0.05) -> float:
    drawdown = compute_drawdown(result_df)
    return float(np.percentile(drawdown, pct * 100))


def worst_n_drawdowns(result_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    episodes = sorted(extract_drawdown_episodes(result_df), key=lambda ep: ep.max_drawdown)[:n]
    return pd.DataFrame([
        {"start": ep.start, "end": ep.end, "max_drawdown": ep.max_drawdown, "n_days": ep.n_days}
        for ep in episodes
    ])


def sharpe_ratio(result_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    pnl = result_df["pnl"]
    if pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * math.sqrt(periods_per_year))


def sortino_ratio(result_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    pnl = result_df["pnl"]
    downside = pnl[pnl < 0]
    if len(downside) < 2 or downside.std() == 0:
        return float("nan")
    return float(pnl.mean() / downside.std() * math.sqrt(periods_per_year))


def annualized_pnl(result_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    return float(result_df["pnl"].mean() * periods_per_year)


def calmar_ratio(result_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    max_dd = max_drawdown(result_df)
    if max_dd == 0:
        return float("nan")
    return annualized_pnl(result_df, periods_per_year) / abs(max_dd)


def trade_win_loss_stats(trade_events: pd.DataFrame, portfolio: "Portfolio") -> dict:
    """Per CLOSED clip (not per leg): sum realized_pnl across its legs; the
    sign of that sum determines win vs loss."""
    closed = trade_events[trade_events["is_closed"]]
    wins = losses = 0
    for clip_id in closed["clip_id"]:
        legs = [pos for pos in portfolio.positions if pos.clip_id == clip_id]
        clip_pnl = sum(leg.realized_pnl or 0.0 for leg in legs)
        if clip_pnl >= 0:
            wins += 1
        else:
            losses += 1
    n_closed = wins + losses
    return {
        "trades_opened": len(trade_events),
        "trades_closed": n_closed,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wins / n_closed if n_closed > 0 else None,
        "loss_rate": losses / n_closed if n_closed > 0 else None,
    }


def best_worst_day(result_df: pd.DataFrame) -> dict:
    best_idx = result_df["pnl"].idxmax()
    worst_idx = result_df["pnl"].idxmin()
    best_pnl = float(result_df["pnl"].loc[best_idx])
    worst_pnl = float(result_df["pnl"].loc[worst_idx])
    return {
        "best_day_date": result_df["date"].loc[best_idx],
        "best_day_pnl": best_pnl,
        "worst_day_date": result_df["date"].loc[worst_idx],
        "worst_day_pnl": worst_pnl,
        "best_worst_ratio": best_pnl / worst_pnl if worst_pnl != 0 else float("nan"),
    }


def annual_pnl_and_drawdown(result_df: pd.DataFrame) -> pd.DataFrame:
    df = result_df[["date", "pnl"]].copy()
    df["year"] = df["date"].dt.year
    df["drawdown"] = compute_drawdown(result_df).values
    grouped = df.groupby("year").agg(pnl=("pnl", "sum"), drawdown=("drawdown", "min"))
    return grouped.reset_index()
