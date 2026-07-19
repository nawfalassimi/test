from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fxbacktest.analytics.blotter import build_trade_blotter
from fxbacktest.analytics.report import build_report
from fxbacktest.analytics.trades import extract_trade_events
from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.execution.transaction_costs import OptionCostSpec, PairCostSpec, SpotCostSpec, TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.short_vol_carry import ShortVolCarryStrategy


@pytest.fixture(scope="module")
def backtest_outputs():
    df = SyntheticFxDataGenerator(start="2022-01-01", end="2022-06-30", seed=8).generate()
    pricer = GarmanKohlhagenPricer()
    strategy = ShortVolCarryStrategy()
    hedger = DailyDeltaHedger(pricer, mode="daily")
    cost_model = TransactionCostModel(by_pair={
        "EURUSD": PairCostSpec(option=OptionCostSpec(kind="vol_spread", vol_spread_bp=50.0),
                              spot=SpotCostSpec(spread_pips=1.0)),
    })
    result_df, portfolio = run_backtest(df, strategy, hedger, pricer, cost_model=cost_model)
    trade_events = extract_trade_events(portfolio)
    blotter = build_trade_blotter(portfolio, df, pricer)
    return result_df, portfolio, trade_events, blotter


def test_build_report_produces_self_contained_html(backtest_outputs, tmp_path):
    result_df, portfolio, trade_events, blotter = backtest_outputs
    config = {"pair": "EURUSD", "strategy": {"name": "short_vol_carry_1m"}}
    out_path = tmp_path / "report.html"

    returned_path = build_report(result_df, portfolio, trade_events, blotter, config, out_path)
    assert returned_path == out_path
    assert out_path.exists()

    html = out_path.read_text(encoding="utf-8")
    assert "Cumulative P&L" in html
    assert "Cumulative P&L (with friction)" in html
    # the parenthetical detail was intentionally dropped from the chart
    # title/legend to leave more room for the plot itself
    assert "before transaction costs" not in html
    assert "after transaction costs" not in html
    assert "Entries" in html
    assert "Exits" in html
    assert "Trade blotter" in html
    assert "Sharpe ratio" in html
    assert "Worst 5 drawdowns" in html
    assert "plotly" in html.lower()
    assert len(html) > 100_000  # inlined plotly.js alone is a few MB; sanity floor

    # a matured option's current_vol is None (T<=0 at expiry) — this must
    # serialize as JSON `null`, not a bare `NaN` token (which pandas can
    # silently reintroduce via numeric-dtype coercion, and which renders as
    # the literal text "NaN" in the embedded blotter table's JS). Scoped to
    # just the blotter's own embedded array, since Plotly's chart data (e.g.
    # the monthly heatmap's incomplete months) legitimately embeds NaN too.
    match = re.search(r"const allRows = (\[.*?\]);", html, re.DOTALL)
    assert match is not None
    blotter_records = json.loads(match.group(1))
    assert any(r["status"] == "exit" for r in blotter_records)
    assert all(r["current_vol"] is None for r in blotter_records if r["status"] == "exit")


def test_build_report_uses_two_column_tearsheet_layout(backtest_outputs, tmp_path):
    result_df, portfolio, trade_events, blotter = backtest_outputs
    config = {"pair": "EURUSD", "strategy": {"name": "short_vol_carry_1m"}}
    out_path = tmp_path / "report_layout.html"

    build_report(result_df, portfolio, trade_events, blotter, config, out_path)
    html = out_path.read_text(encoding="utf-8")

    assert 'class="container"' in html
    assert 'id="left"' in html
    assert 'id="right"' in html
    assert 'class="full-width-section"' in html
    # the left column must appear before the right column in source order
    assert html.index('id="left"') < html.index('id="right"')
