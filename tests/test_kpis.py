from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fxbacktest.analytics.kpis import (
    extract_drawdown_episodes,
    sharpe_ratio,
    sortino_ratio,
    trade_win_loss_stats,
)
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position


def _result_df(pnls):
    dates = pd.bdate_range("2022-01-03", periods=len(pnls))
    return pd.DataFrame({"date": dates, "pnl": pnls, "friction_cost": [0.0] * len(pnls)})


def test_extract_drawdown_episodes_matches_hand_built_shape():
    pnls = [100, -50, -30, 90, -200, 0, 300, -10, 0, 0]
    # cum_pnl: 100, 50, 20, 110, -90, -90, 210, 200, 200, 200
    result_df = _result_df(pnls)
    dates = result_df["date"]

    episodes = extract_drawdown_episodes(result_df)
    assert len(episodes) == 3

    ep1, ep2, ep3 = episodes
    assert ep1.start == dates.iloc[0] and ep1.end == dates.iloc[3]
    assert ep1.max_drawdown == pytest.approx(-80.0)
    assert ep1.n_days == (dates.iloc[3] - dates.iloc[0]).days

    assert ep2.start == dates.iloc[3] and ep2.end == dates.iloc[6]
    assert ep2.max_drawdown == pytest.approx(-200.0)

    # trailing/ongoing episode: never recovers by the last available date
    assert ep3.start == dates.iloc[6] and ep3.end == dates.iloc[9]
    assert ep3.max_drawdown == pytest.approx(-10.0)


def test_sharpe_and_sortino_match_independent_hand_calculation():
    pnls = [10, -5, 20, -15, 5]
    result_df = _result_df(pnls)

    arr = np.array(pnls, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    expected_sharpe = mean / std * math.sqrt(252)

    downside = arr[arr < 0]
    downside_std = downside.std(ddof=1)
    expected_sortino = mean / downside_std * math.sqrt(252)

    assert sharpe_ratio(result_df) == pytest.approx(expected_sharpe)
    assert sortino_ratio(result_df) == pytest.approx(expected_sortino)


def _closed_clip(clip_id, realized_pnls):
    entry = pd.Timestamp("2022-01-03")
    exit_ = pd.Timestamp("2022-02-02")
    legs = []
    for i, pnl in enumerate(realized_pnls):
        option = FxVanillaOption(pair="EURUSD", strike=1.10, expiry=exit_,
                                 option_type="call" if i == 0 else "put",
                                 notional=1_000_000, trade_date=entry)
        legs.append(Position(instrument=option, qty=-1.0, clip_id=clip_id, strategy_id="s",
                             entry_date=entry, entry_price=1000.0,
                             is_open=False, exit_date=exit_, exit_price=500.0, realized_pnl=pnl))
    return legs


def test_trade_win_loss_stats_counts_wins_and_losses_per_clip():
    positions = (
        _closed_clip("clip_win_1", [300.0, 100.0])   # sum = 400 -> win
        + _closed_clip("clip_win_2", [50.0, -20.0])  # sum = 30 -> win
        + _closed_clip("clip_loss_1", [-200.0, -50.0])  # sum = -250 -> loss
    )
    # a still-open clip should not count toward closed win/loss stats
    open_leg = Position(
        instrument=FxVanillaOption(pair="EURUSD", strike=1.10, expiry=pd.Timestamp("2022-03-01"),
                                   option_type="call", notional=1_000_000,
                                   trade_date=pd.Timestamp("2022-02-01")),
        qty=-1.0, clip_id="clip_open", strategy_id="s",
        entry_date=pd.Timestamp("2022-02-01"), entry_price=1000.0,
    )
    portfolio = Portfolio(positions=positions + [open_leg])

    trade_events = pd.DataFrame([
        {"clip_id": "clip_win_1", "is_closed": True},
        {"clip_id": "clip_win_2", "is_closed": True},
        {"clip_id": "clip_loss_1", "is_closed": True},
        {"clip_id": "clip_open", "is_closed": False},
    ])

    stats = trade_win_loss_stats(trade_events, portfolio)
    assert stats["trades_opened"] == 4
    assert stats["trades_closed"] == 3
    assert stats["winning_trades"] == 2
    assert stats["losing_trades"] == 1
    assert stats["win_rate"] == pytest.approx(2 / 3)
