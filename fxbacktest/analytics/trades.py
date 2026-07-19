from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, List

import pandas as pd

if TYPE_CHECKING:
    from fxbacktest.portfolio.portfolio import Portfolio


def extract_trade_events(portfolio: "Portfolio",
                          exclude_strategy_ids: Iterable[str] = ("hedge",)) -> pd.DataFrame:
    """One row per clip_id (a strategy "trade", which may span multiple legs —
    e.g. a straddle's call + put — opened together). entry_date is the
    earliest entry_date across the clip's legs; exit_date is the latest exit
    date across legs if ALL legs are closed, else None (still open).

    Clips whose strategy_id is in exclude_strategy_ids (default: the delta
    hedger's daily spot clips) are dropped — hedge rebalancing trades are not
    "strategy trades" for entry/exit marker purposes.
    """
    by_clip: dict = defaultdict(list)
    for pos in portfolio.positions:
        by_clip[pos.clip_id].append(pos)

    exclude = set(exclude_strategy_ids)
    rows = []
    for clip_id, legs in by_clip.items():
        strategy_ids = {leg.strategy_id for leg in legs}
        assert len(strategy_ids) == 1, f"clip {clip_id} mixes strategy_ids: {strategy_ids}"
        strategy_id = legs[0].strategy_id
        if strategy_id in exclude:
            continue

        is_closed = all(not leg.is_open for leg in legs)
        exit_date = max(leg.exit_date for leg in legs) if is_closed else None

        rows.append({
            "clip_id": clip_id,
            "strategy_id": strategy_id,
            "entry_date": min(leg.entry_date for leg in legs),
            "exit_date": exit_date,
            "is_closed": is_closed,
        })

    return pd.DataFrame(rows, columns=["clip_id", "strategy_id", "entry_date", "exit_date", "is_closed"])
