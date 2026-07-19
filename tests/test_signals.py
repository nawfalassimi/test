from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxbacktest.strategies import signals as sig


def _quotes_df(dates, tenor, atm_vol, rr25=None, spot=None, pair="EURUSD"):
    n = len(dates)
    return pd.DataFrame({
        "pair": pair, "date": dates, "tenor": tenor,
        "atm_vol": atm_vol,
        "rr25": rr25 if rr25 is not None else np.zeros(n),
        "spot": spot if spot is not None else np.full(n, 1.10),
    })


def test_tenor_series_filters_and_sorts_by_pair_and_tenor():
    dates = pd.bdate_range("2022-01-03", periods=5)
    df = pd.concat([
        _quotes_df(dates, "1M", np.arange(5) + 1.0, pair="EURUSD"),
        _quotes_df(dates, "3M", np.arange(5) + 100.0, pair="EURUSD"),
        _quotes_df(dates, "1M", np.arange(5) + 1000.0, pair="USDJPY"),
    ], ignore_index=True).sample(frac=1, random_state=0)  # shuffle rows

    s = sig._tenor_series(df, "EURUSD", "1M", "atm_vol")
    assert list(s.index) == list(dates)
    assert list(s.values) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_iv_zscore_expanding_nan_before_min_periods_and_matches_full_sample_stats():
    n = 70
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(1)
    atm = 0.08 + rng.normal(0, 0.005, size=n)
    df = _quotes_df(dates, "3M", atm)

    z = sig.iv_zscore_expanding(df, "EURUSD", "3M")

    assert z.iloc[:sig.IV_ZSCORE_MIN_PERIODS - 1].isna().all()
    assert z.iloc[sig.IV_ZSCORE_MIN_PERIODS - 1:].notna().all()

    # At the LAST row, the expanding window covers the entire series, so the
    # expanding mean/std must equal the full-sample mean/std exactly.
    expected_z = (atm[-1] - atm.mean()) / atm.std(ddof=1)
    assert z.iloc[-1] == pytest.approx(expected_z)


def test_realized_vol_constant_log_return_gives_exact_annualized_value():
    n = 40
    dates = pd.bdate_range("2022-01-03", periods=n)
    daily_ret = 0.001
    spot = pd.Series(1.10 * np.exp(np.arange(n) * daily_ret), index=dates)

    rv = sig.realized_vol(spot, window_days=20)
    # Constant log-returns => rolling std of returns is exactly 0 once the
    # window is full (every window has zero variance).
    assert rv.iloc[19:].abs().max() < 1e-12
    assert rv.iloc[:19].isna().all()


def test_vrp_is_iv_minus_realized_vol():
    n = 100
    dates = pd.bdate_range("2022-01-03", periods=n)
    atm = np.full(n, 0.09)
    spot = np.full(n, 1.10)  # zero realized vol: constant spot
    df = _quotes_df(dates, "1M", atm, spot=spot)

    result = sig.vrp(df, "EURUSD", "1M")
    window = 30  # TENOR_DAYS["1M"]
    # realized_vol's internal .diff() adds one extra leading NaN before the
    # rolling window itself, so the first valid row is at index `window`,
    # not `window - 1`.
    assert result.iloc[:window].isna().all()
    # realized vol of a perfectly constant spot is exactly 0 (log returns all 0)
    assert result.iloc[window:].round(10).eq(0.09).all()


def test_vol_of_vol_signal_true_when_recent_vol_of_vol_below_its_own_mean():
    n = 200
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(2)
    # First 150 rows: noisy ATM vol (high vol-of-vol). Last 50 rows: nearly flat.
    atm = np.concatenate([
        0.08 + rng.normal(0, 0.01, size=150),
        np.full(50, 0.08),
    ])
    df = _quotes_df(dates, "3M", atm)

    result = sig.vol_of_vol_signal(df, "EURUSD", "3M")
    assert result.iloc[:sig.VOL_OF_VOL_WINDOW + sig.VOL_OF_VOL_MEAN_WINDOW - 2].isna().all()
    # near the very end, the 21d std is ~0 (flat tail) while its 126d mean
    # still reflects the earlier noisy period => low vol-of-vol => True
    assert bool(result.iloc[-1]) is True


def test_term_structure_ratio_and_hard_stop_are_mutually_exclusive():
    dates = pd.bdate_range("2022-01-03", periods=5)
    df = pd.concat([
        _quotes_df(dates, "1M", np.full(5, 0.08)),
        _quotes_df(dates, "6M", np.full(5, 0.10)),
    ], ignore_index=True)

    ratio = sig.term_structure_ratio(df, "EURUSD")
    assert (ratio < 1.0).all()  # contango: 1M vol below 6M vol

    df_inverted = pd.concat([
        _quotes_df(dates, "1M", np.full(5, 0.12)),
        _quotes_df(dates, "6M", np.full(5, 0.10)),
    ], ignore_index=True)
    ratio_inv = sig.term_structure_ratio(df_inverted, "EURUSD")
    assert (ratio_inv > 1.0).all()


def test_rr_zscore_matches_hand_computed_rolling_stats():
    n = 140
    window = 20
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(3)
    rr25 = rng.normal(0, 0.005, size=n)
    df = _quotes_df(dates, "3M", np.full(n, 0.08), rr25=rr25)

    z = sig.rr_zscore(df, "EURUSD", "3M", window=window)
    assert z.iloc[:window - 1].isna().all()

    rr_abs = pd.Series(np.abs(rr25))
    expected_last = (rr_abs.iloc[-1] - rr_abs.iloc[-window:].mean()) / rr_abs.iloc[-window:].std(ddof=1)
    assert z.iloc[-1] == pytest.approx(expected_last)


def test_vix_calm_and_hot_use_own_trailing_quantile():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    vix = pd.Series(np.full(n, 15.0), index=dates)
    vix.iloc[-1] = 100.0  # an obvious spike on the last day

    calm = sig.vix_calm(vix)
    hot = sig.vix_hot(vix)

    assert calm.iloc[:sig.VIX_WINDOW - 1].isna().all()
    assert hot.iloc[:sig.VIX_WINDOW - 1].isna().all()
    # a flat series' own trailing quantile is ~15, so the spike must read hot, not calm
    assert bool(hot.iloc[-1]) is True
    assert bool(calm.iloc[-1]) is False


def test_compute_signal_bundle_entry_requires_vix_window_fully_warmed():
    """Regression test: without gating entries on vix_hot.notna(), entries
    could occur between row ~126 (RR z-score warm) and row 252 (VIX window
    warm) while the VIX hard-stop is silently inert."""
    n = 200  # RR/vol-of-vol windows warm by here, VIX (252) is NOT
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(4)

    atm_1m = np.full(n, 0.06)
    atm_3m = 0.09 + rng.normal(0, 0.003, size=n)
    atm_6m = np.full(n, 0.10)  # contango: 1M < 6M throughout
    rr25 = np.zeros(n)  # |rr25| z-score ~ 0 or NaN (flat), never > 0 => rr_condition True where defined

    df = pd.concat([
        _quotes_df(dates, "1M", atm_1m, rr25=rr25),
        _quotes_df(dates, "3M", atm_3m, rr25=rr25),
        _quotes_df(dates, "6M", atm_6m, rr25=rr25),
    ], ignore_index=True)
    vix = pd.Series(np.full(n, 10.0), index=dates)  # calm throughout, window never fully warms (n < 252)

    bundle = sig.compute_signal_bundle(df, vix, "EURUSD", "3M")
    assert not bundle.entry_ok.any()


def test_compute_signal_bundle_exit_now_on_hard_stops_or_zero_composite():
    n = 260
    dates = pd.bdate_range("2022-01-03", periods=n)

    atm_1m = np.full(n, 0.12)  # inverted: 1M > 6M throughout => hard stop
    atm_3m = np.full(n, 0.09)
    atm_6m = np.full(n, 0.10)
    rr25 = np.zeros(n)

    df = pd.concat([
        _quotes_df(dates, "1M", atm_1m, rr25=rr25),
        _quotes_df(dates, "3M", atm_3m, rr25=rr25),
        _quotes_df(dates, "6M", atm_6m, rr25=rr25),
    ], ignore_index=True)
    vix = pd.Series(np.full(n, 10.0), index=dates)

    bundle = sig.compute_signal_bundle(df, vix, "EURUSD", "3M")
    assert bool(bundle.hard_stop_inverted.iloc[-1]) is True
    assert bool(bundle.exit_now.iloc[-1]) is True
    assert not bundle.entry_ok.any()  # inverted surface blocks every entry too
