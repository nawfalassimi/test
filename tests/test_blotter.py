from __future__ import annotations

import pandas as pd
import pytest

from fxbacktest.analytics.blotter import build_trade_blotter
from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.instruments.option import FxVanillaOption
from fxbacktest.instruments.spot import FxSpot
from fxbacktest.portfolio.portfolio import Portfolio
from fxbacktest.portfolio.position import Position
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer


@pytest.fixture(scope="module")
def quotes_df():
    return SyntheticFxDataGenerator(start="2022-01-01", end="2022-02-10", seed=6).generate()


@pytest.fixture(scope="module")
def dates(quotes_df):
    return sorted(quotes_df["date"].drop_duplicates().tolist())


def test_blotter_row_counts_match_position_date_spans(quotes_df, dates):
    entry_date, exit_date, last_date = dates[0], dates[15], dates[-1]
    # option expiry set well after our hand-picked exit_date so T > 0 throughout
    # the window (isolates the row-count/status assertions from the separate
    # "vol goes to None at true expiry" behavior, tested below).
    far_expiry = dates[-1] + pd.Timedelta(days=60)

    call = FxVanillaOption(pair="EURUSD", strike=1.10, expiry=far_expiry, option_type="call",
                           notional=1_000_000, trade_date=entry_date)
    put = FxVanillaOption(pair="EURUSD", strike=1.10, expiry=far_expiry, option_type="put",
                          notional=1_000_000, trade_date=entry_date)
    call_pos = Position(instrument=call, qty=-1.0, clip_id="clip1", strategy_id="s",
                        entry_date=entry_date, entry_price=10_000.0,
                        is_open=False, exit_date=exit_date, exit_price=9_000.0)
    put_pos = Position(instrument=put, qty=-1.0, clip_id="clip1", strategy_id="s",
                       entry_date=entry_date, entry_price=9_500.0,
                       is_open=False, exit_date=exit_date, exit_price=8_500.0)

    spot = FxSpot(pair="EURUSD", notional=50_000.0)
    spot_pos = Position(instrument=spot, qty=1.0, clip_id="desk1", strategy_id="manual_desk",
                        entry_date=entry_date, entry_price=55_000.0)  # stays open

    portfolio = Portfolio(positions=[call_pos, put_pos, spot_pos])
    pricer = GarmanKohlhagenPricer()

    blotter = build_trade_blotter(portfolio, quotes_df, pricer)

    n_dates_option_window = len([d for d in dates if entry_date <= d <= exit_date])
    n_dates_spot_window = len([d for d in dates if entry_date <= d <= last_date])
    expected_rows = 2 * n_dates_option_window + n_dates_spot_window
    assert len(blotter) == expected_rows

    for trade_id in blotter["trade_id"].unique():
        rows = blotter[blotter["trade_id"] == trade_id]
        assert (rows["status"] == "new").sum() == 1
        assert rows.loc[rows["status"] == "new", "date"].iloc[0] == entry_date

    for trade_id in ["clip1_call", "clip1_put"]:
        rows = blotter[blotter["trade_id"] == trade_id]
        assert (rows["status"] == "exit").sum() == 1
        assert rows.loc[rows["status"] == "exit", "date"].iloc[0] == exit_date

    spot_rows = blotter[blotter["trade_id"] == "desk1_spot"]
    assert (spot_rows["status"] == "exit").sum() == 0  # still open, never exits
    assert spot_rows["current_vol"].isna().all()


def test_hedge_positions_included_in_blotter_by_default(quotes_df, dates):
    entry_date = dates[0]
    spot_pos = Position(
        instrument=FxSpot(pair="EURUSD", notional=50_000.0), qty=1.0,
        clip_id="hedge_20220103", strategy_id="hedge", entry_date=entry_date, entry_price=55_000.0,
    )
    portfolio = Portfolio(positions=[spot_pos])
    pricer = GarmanKohlhagenPricer()

    blotter = build_trade_blotter(portfolio, quotes_df, pricer)
    assert len(blotter) > 0
    assert (blotter["trade_id"] == "hedge_20220103_spot").all()


def test_exclude_strategy_ids_is_overridable_in_blotter(quotes_df, dates):
    entry_date = dates[0]
    spot_pos = Position(
        instrument=FxSpot(pair="EURUSD", notional=50_000.0), qty=1.0,
        clip_id="hedge_20220103", strategy_id="hedge", entry_date=entry_date, entry_price=55_000.0,
    )
    portfolio = Portfolio(positions=[spot_pos])
    pricer = GarmanKohlhagenPricer()

    blotter = build_trade_blotter(portfolio, quotes_df, pricer, exclude_strategy_ids=("hedge",))
    assert len(blotter) == 0


def test_blotter_option_vol_is_none_at_true_expiry(quotes_df, dates):
    entry_date = dates[0]
    expiry = dates[10]  # a real quotes_df date, so we can observe T<=0 there

    call = FxVanillaOption(pair="EURUSD", strike=1.10, expiry=expiry, option_type="call",
                           notional=1_000_000, trade_date=entry_date)
    call_pos = Position(instrument=call, qty=-1.0, clip_id="clip2", strategy_id="s",
                        entry_date=entry_date, entry_price=5_000.0,
                        is_open=False, exit_date=expiry, exit_price=0.0)

    portfolio = Portfolio(positions=[call_pos])
    pricer = GarmanKohlhagenPricer()
    blotter = build_trade_blotter(portfolio, quotes_df, pricer)

    at_expiry = blotter[blotter["date"] == expiry]
    assert len(at_expiry) == 1
    assert pd.isna(at_expiry.iloc[0]["current_vol"])

    before_expiry = blotter[blotter["date"] == dates[5]]
    assert len(before_expiry) == 1
    assert before_expiry.iloc[0]["current_vol"] is not None
    assert not pd.isna(before_expiry.iloc[0]["current_vol"])
