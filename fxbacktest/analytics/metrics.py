from __future__ import annotations

import pandas as pd


def compute_cum_pnl(result: pd.DataFrame) -> pd.Series:
    return result["pnl"].cumsum()


def compute_drawdown(result: pd.DataFrame) -> pd.Series:
    cum_pnl = compute_cum_pnl(result)
    running_max = cum_pnl.cummax()
    return cum_pnl - running_max


def max_drawdown(result: pd.DataFrame) -> float:
    return compute_drawdown(result).min()
