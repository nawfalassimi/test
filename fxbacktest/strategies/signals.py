from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from fxbacktest.data.schema import TENOR_DAYS

IV_ZSCORE_MIN_PERIODS = 60
IV_ZSCORE_THRESHOLD = 0.5
RR_ZSCORE_WINDOW = 126
VOL_OF_VOL_WINDOW = 21
VOL_OF_VOL_MEAN_WINDOW = 126
VIX_WINDOW = 252
VIX_CALM_PCTL = 0.40
VIX_HOT_PCTL = 0.90
CONTANGO_SHORT_TENOR = "1M"
CONTANGO_LONG_TENOR = "6M"


def _tenor_series(quotes_df: pd.DataFrame, pair: str, tenor: str, column: str) -> pd.Series:
    """Date-indexed, date-sorted series of `column` for one (pair, tenor)."""
    sub = quotes_df[(quotes_df["pair"] == pair) & (quotes_df["tenor"] == tenor)]
    return sub.sort_values("date").set_index("date")[column]


def iv_zscore_expanding(quotes_df: pd.DataFrame, pair: str, tenor: str) -> pd.Series:
    """Signal #1 input. Expanding (not rolling) mean/std of ATM vol at the
    strategy's own traded tenor — unlike the RR/VIX signals below, the window
    grows without bound rather than sliding, so early history keeps
    contributing to the baseline forever."""
    atm = _tenor_series(quotes_df, pair, tenor, "atm_vol")
    mean = atm.expanding(min_periods=IV_ZSCORE_MIN_PERIODS).mean()
    std = atm.expanding(min_periods=IV_ZSCORE_MIN_PERIODS).std()
    return (atm - mean) / std


def realized_vol(spot: pd.Series, window_days: int) -> pd.Series:
    log_ret = np.log(spot).diff()
    return log_ret.rolling(window_days, min_periods=window_days).std() * np.sqrt(252)


def vrp(quotes_df: pd.DataFrame, pair: str, tenor: str) -> pd.Series:
    """Signal #2 input: same-tenor ATM IV minus RV of spot over a MATCHING
    window (TENOR_DAYS[tenor] rows, e.g. 91 for 3M) — a calendar tenor length
    used directly as a row count on daily/business-day data."""
    atm = _tenor_series(quotes_df, pair, tenor, "atm_vol")
    spot = _tenor_series(quotes_df, pair, tenor, "spot")
    rv = realized_vol(spot, TENOR_DAYS[tenor])
    return atm - rv


def vol_of_vol_signal(quotes_df: pd.DataFrame, pair: str, tenor: str) -> pd.Series:
    """Signal #4 (already boolean: True == 'low vol-of-vol'). 21d rolling std
    of ATM vol, compared to ITS OWN 126d rolling mean."""
    atm = _tenor_series(quotes_df, pair, tenor, "atm_vol")
    vov = atm.rolling(VOL_OF_VOL_WINDOW, min_periods=VOL_OF_VOL_WINDOW).std()
    vov_mean = vov.rolling(VOL_OF_VOL_MEAN_WINDOW, min_periods=VOL_OF_VOL_MEAN_WINDOW).mean()
    # `<` against a NaN operand evaluates to False, not NaN, in pandas/numpy —
    # .where() restores a genuine NaN during warm-up so callers (compute_signal_bundle's
    # warmed_up gate) can distinguish "not yet defined" from "actually False."
    return (vov < vov_mean).where(vov_mean.notna())


def term_structure_ratio(quotes_df: pd.DataFrame, pair: str) -> pd.Series:
    """ATM(1M)/ATM(6M), ALWAYS at fixed 1M/6M pillars regardless of the
    strategy's traded tenor. Single source of truth for both signal #5
    (ratio < 1) and the hard-stop (ratio > 1)."""
    atm_1m = _tenor_series(quotes_df, pair, CONTANGO_SHORT_TENOR, "atm_vol")
    atm_6m = _tenor_series(quotes_df, pair, CONTANGO_LONG_TENOR, "atm_vol")
    return atm_1m / atm_6m


def rr_zscore(quotes_df: pd.DataFrame, pair: str, tenor: str, window: int = RR_ZSCORE_WINDOW) -> pd.Series:
    """Mandatory entry condition input: rolling (not expanding) z-score of
    |rr25|, at the strategy's own traded tenor. Reads the raw rr25 column
    directly off quotes_df (not reconstructed from the smile spline, which is
    only exact at pillars)."""
    rr_abs = _tenor_series(quotes_df, pair, tenor, "rr25").abs()
    mean = rr_abs.rolling(window, min_periods=window).mean()
    std = rr_abs.rolling(window, min_periods=window).std()
    return (rr_abs - mean) / std


def vix_calm(vix: pd.Series, pctl: float = VIX_CALM_PCTL, window: int = VIX_WINDOW) -> pd.Series:
    """Signal #3 (True == calm): current value below the trailing window's
    own pctl-quantile. See vol_of_vol_signal for why `.where()` is needed to
    keep warm-up NaN instead of a comparison-against-NaN False."""
    threshold = vix.rolling(window, min_periods=window).quantile(pctl)
    return (vix < threshold).where(threshold.notna())


def vix_hot(vix: pd.Series, pctl: float = VIX_HOT_PCTL, window: int = VIX_WINDOW) -> pd.Series:
    """Hard-stop input (True == breach). Same rolling-quantile technique as
    vix_calm, mirrored at the 90th percentile."""
    threshold = vix.rolling(window, min_periods=window).quantile(pctl)
    return (vix > threshold).where(threshold.notna())


class SignalBundle(NamedTuple):
    composite_flags: pd.DataFrame   # 5 bool columns: iv_z, vrp, vix_calm, vol_of_vol, contango
    composite_count: pd.Series      # int 0..5
    rr_condition: pd.Series         # bool, the mandatory (separate) entry gate
    hard_stop_inverted: pd.Series   # bool: term-structure ratio > 1 (NOT contango)
    hard_stop_vix: pd.Series        # bool: vix above 90th pctl
    entry_ok: pd.Series             # bool
    exit_now: pd.Series             # bool


def compute_signal_bundle(quotes_df: pd.DataFrame, vix: pd.Series, pair: str, tenor: str) -> SignalBundle:
    idx = _tenor_series(quotes_df, pair, tenor, "atm_vol").index
    vix = vix.reindex(idx)

    z_iv = iv_zscore_expanding(quotes_df, pair, tenor)
    vrp_s = vrp(quotes_df, pair, tenor)
    calm = vix_calm(vix)
    low_vov = vol_of_vol_signal(quotes_df, pair, tenor)
    ratio = term_structure_ratio(quotes_df, pair).reindex(idx)

    flags = pd.DataFrame({
        "iv_z": (z_iv > IV_ZSCORE_THRESHOLD).fillna(False),
        "vrp": (vrp_s > 0).fillna(False),
        "vix_calm": calm.fillna(False),
        "vol_of_vol": low_vov.fillna(False),
        "contango": (ratio < 1.0).fillna(False),
    }, index=idx)
    composite_count = flags.sum(axis=1)

    rr_cond = (rr_zscore(quotes_df, pair, tenor) < 0).fillna(False)
    hard_stop_inverted = (ratio > 1.0).fillna(False)
    vix_hot_s = vix_hot(vix)
    hard_stop_vix = vix_hot_s.fillna(False)

    # Entries additionally require the VIX 252-day window to be fully warmed
    # (vix_hot_s.notna()) — without this, entries could occur between row
    # ~126 (RR z-score warm) and row 252 (VIX window warm) while the VIX
    # hard-stop is silently inert.
    warmed_up = vix_hot_s.notna()

    entry_ok = (composite_count >= 2) & rr_cond & ~hard_stop_inverted & ~hard_stop_vix & warmed_up
    exit_now = hard_stop_inverted | hard_stop_vix | (composite_count == 0)

    return SignalBundle(flags, composite_count, rr_cond, hard_stop_inverted, hard_stop_vix, entry_ok, exit_now)
