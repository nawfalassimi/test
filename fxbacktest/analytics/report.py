from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from fxbacktest.analytics.kpis import (
    annual_pnl_and_drawdown,
    annualized_pnl,
    average_drawdown,
    average_top_n_drawdown,
    best_worst_day,
    calmar_ratio,
    cum_pnl_without_friction,
    drawdown_percentile,
    sharpe_ratio,
    sortino_ratio,
    trade_win_loss_stats,
    worst_n_drawdowns,
)
from fxbacktest.analytics.metrics import compute_cum_pnl, compute_drawdown, max_drawdown

if TYPE_CHECKING:
    from fxbacktest.portfolio.portfolio import Portfolio

# Chart set deliberately excludes: "Currency Exposures" (redundant with the
# Delta chart in this single-pair system), "Funding details" (no funding
# model exists in this codebase — showing fake zeros would look like a bug),
# and "Size ($)" (ambiguous meaning, deferred).

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BLOTTER_DISPLAY_COLUMNS = [
    "trade_id", "strategy_id", "instrument_type", "entry_date", "date", "status",
    "entry_price", "current_price", "entry_vol", "current_vol", "delta", "vega", "gamma",
]


def _fmt_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:,.0f}"


def _fmt_ratio(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}"


def _fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:.1f}%"


def _defer_chart_script(chart_html: str) -> str:
    """Neutralize a fig.to_html(full_html=False, ...) fragment's inline
    <script> tag(s) so they don't execute immediately as the browser parses
    them. Plotly's script calls Plotly.newPlot() the instant it runs,
    measuring its container's width AT THAT MOMENT — if that happens before
    the surrounding flex layout has been fully parsed (e.g. before the
    sidebar later in the document exists), the chart locks in the wrong
    (too-wide) initial size. Retagging the script as inert (an unrecognized
    type, so the browser parses but does not execute it) and re-activating
    it later — once the whole page has been laid out — fixes this without
    needing to parse/re-split Plotly's own HTML structure (which wraps the
    div and script in ways that vary and are easy to break with naive
    string-splitting)."""
    return chart_html.replace("<script>", '<script type="text/plain" class="deferred-plotly">')


def _entry_exit_traces(trade_events: pd.DataFrame, dates: pd.Series, series: pd.Series):
    value_by_date = dict(zip(dates, series))

    entries = trade_events["entry_date"]
    entry_x = [d for d in entries if d in value_by_date]
    entry_y = [value_by_date[d] for d in entry_x]

    exits = trade_events["exit_date"].dropna()
    exit_x = [d for d in exits if d in value_by_date]
    exit_y = [value_by_date[d] for d in exit_x]

    entry_trace = go.Scatter(x=entry_x, y=entry_y, mode="markers", name="Entries",
                             marker=dict(symbol="triangle-up", size=12, color="green"))
    exit_trace = go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exits",
                            marker=dict(symbol="triangle-down", size=12, color="red"))
    return entry_trace, exit_trace


def _pnl_chart_with_markers(title: str, result_df: pd.DataFrame, series: pd.Series,
                             trade_events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["date"], y=series, mode="lines", name=title,
                             line=dict(color="royalblue")))
    entry_trace, exit_trace = _entry_exit_traces(trade_events, result_df["date"], series)
    fig.add_trace(entry_trace)
    fig.add_trace(exit_trace)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
                      template="plotly_white", hovermode="x unified")
    return fig


def _daily_pnl_chart(result_df: pd.DataFrame) -> go.Figure:
    rolling = result_df["pnl"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result_df["date"], y=result_df["pnl"], name="Daily P&L", marker_color="royalblue"))
    fig.add_trace(go.Scatter(x=result_df["date"], y=rolling, mode="lines", name="30d rolling mean",
                             line=dict(color="red", dash="dash")))
    fig.update_layout(title="Daily P&L", xaxis_title="Date", yaxis_title="Daily P&L ($)",
                      template="plotly_white")
    return fig


def _drawdown_chart(result_df: pd.DataFrame) -> go.Figure:
    drawdown = compute_drawdown(result_df)
    avg_dd = average_drawdown(result_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["date"], y=drawdown, mode="lines", name="Drawdown",
                             fill="tozeroy", line=dict(color="royalblue")))
    fig.add_hline(y=avg_dd, line_dash="dash", line_color="red", annotation_text="Average drawdown")
    fig.update_layout(title="Drawdown (underwater)", xaxis_title="Date", yaxis_title="Drawdown ($)",
                      template="plotly_white")
    return fig


def _greek_chart(result_df: pd.DataFrame, column: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["date"], y=result_df[column], mode="lines", name=title,
                             line=dict(color="royalblue")))
    fig.update_layout(title=f"{title} over time", xaxis_title="Date", yaxis_title=f"{title} ($)",
                      template="plotly_white")
    return fig


def _monthly_heatmap(result_df: pd.DataFrame) -> go.Figure:
    df = result_df[["date", "pnl"]].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.groupby(["year", "month"])["pnl"].sum().unstack("month").reindex(columns=range(1, 13))
    text = [[("" if pd.isna(v) else f"{v:,.0f}") for v in row] for row in pivot.values]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=MONTH_LABELS, y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0, text=text, texttemplate="%{text}", hoverongaps=False,
    ))
    fig.update_layout(title="Monthly P&L", template="plotly_white")
    return fig


def _annual_bar_chart(result_df: pd.DataFrame) -> go.Figure:
    annual = annual_pnl_and_drawdown(result_df)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=annual["year"], y=annual["pnl"], name="P&L", marker_color="royalblue"))
    fig.add_trace(go.Bar(x=annual["year"], y=annual["drawdown"], name="Drawdown", marker_color="firebrick"))
    fig.update_layout(title="Annual P&L and Drawdown", barmode="group", xaxis_title="Year",
                      yaxis_title="Amount ($)", template="plotly_white")
    return fig


def _kpi_table_html(result_df: pd.DataFrame, trade_events: pd.DataFrame, portfolio: "Portfolio") -> str:
    pnl_with_friction = float(result_df["cum_pnl"].iloc[-1])
    pnl_without_friction = float(cum_pnl_without_friction(result_df).iloc[-1])
    total_friction_cost = float(result_df["friction_cost"].sum())
    win_loss = trade_win_loss_stats(trade_events, portfolio)
    bw = best_worst_day(result_df)

    rows = [
        ("P&L with friction costs", _fmt_money(pnl_with_friction)),
        ("P&L without friction costs", _fmt_money(pnl_without_friction)),
        ("Total friction costs", _fmt_money(total_friction_cost)),
        ("Annualized P&L", _fmt_money(annualized_pnl(result_df))),
        ("Annualized P&L / Max Drawdown (Calmar)", _fmt_ratio(calmar_ratio(result_df))),
        ("Sharpe ratio", _fmt_ratio(sharpe_ratio(result_df))),
        ("Sortino ratio", _fmt_ratio(sortino_ratio(result_df))),
        ("Strategy trades opened", str(win_loss["trades_opened"])),
        ("Strategy trades closed", str(win_loss["trades_closed"])),
        ("Winning trades", f"{win_loss['winning_trades']} ({_fmt_pct(win_loss['win_rate'])})"),
        ("Losing trades", f"{win_loss['losing_trades']} ({_fmt_pct(win_loss['loss_rate'])})"),
        ("Maximum drawdown", _fmt_money(max_drawdown(result_df))),
        ("Average drawdown", _fmt_money(average_drawdown(result_df))),
        ("Average of 5 worst drawdowns", _fmt_money(average_top_n_drawdown(result_df, 5))),
        ("5th percentile drawdown", _fmt_money(drawdown_percentile(result_df, 0.05))),
        ("Best day", f"{_fmt_money(bw['best_day_pnl'])} ({bw['best_day_date'].date()})"),
        ("Worst day", f"{_fmt_money(bw['worst_day_pnl'])} ({bw['worst_day_date'].date()})"),
        ("Best / worst day P&L ratio", _fmt_ratio(bw["best_worst_ratio"])),
    ]
    body = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="kpi-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{body}</tbody></table>'


def _worst_drawdowns_html(result_df: pd.DataFrame, n: int = 5) -> str:
    table = worst_n_drawdowns(result_df, n)
    if table.empty:
        return "<p>No drawdown episodes.</p>"
    body = "".join(
        f"<tr><td>{row.start.date()}</td><td>{row.end.date()}</td>"
        f"<td>{_fmt_money(row.max_drawdown)}</td><td>{row.n_days}</td></tr>"
        for row in table.itertuples()
    )
    return (
        '<table class="kpi-table"><thead><tr><th>Start</th><th>End</th>'
        f"<th>Max Drawdown</th><th>Nb Days</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _blotter_section_html(blotter: pd.DataFrame) -> str:
    display = blotter[BLOTTER_DISPLAY_COLUMNS].copy() if not blotter.empty else pd.DataFrame(columns=BLOTTER_DISPLAY_COLUMNS)
    for col in ("entry_date", "date"):
        display[col] = display[col].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "")
    for col in ("entry_price", "current_price", "delta", "vega", "gamma"):
        display[col] = display[col].apply(lambda v: None if pd.isna(v) else round(float(v), 2))
    for col in ("entry_vol", "current_vol"):
        display[col] = display[col].apply(lambda v: None if pd.isna(v) else round(float(v), 4))

    records = display.to_dict(orient="records")
    # pandas silently coerces None back to NaN in a numeric-dtype column
    # (e.g. after the .apply() above), so json.dumps would otherwise emit a
    # bare `NaN` token — valid as a JS literal but not as JSON, and it renders
    # as the literal text "NaN" in the table instead of an empty cell.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    data_json = json.dumps(records)
    columns_json = json.dumps(BLOTTER_DISPLAY_COLUMNS)

    return f"""
    <div id="blotter-section">
      <input id="blotter-search" type="text" placeholder="Filter rows (any column)..." />
      <span id="blotter-page-info"></span>
      <button id="blotter-prev">Prev</button>
      <button id="blotter-next">Next</button>
      <table class="blotter-table" id="blotter-table">
        <thead><tr id="blotter-head"></tr></thead>
        <tbody id="blotter-body"></tbody>
      </table>
    </div>
    <script>
    (function() {{
      const columns = {columns_json};
      const allRows = {data_json};
      const pageSize = 50;
      let page = 0;
      let filtered = allRows;

      const head = document.getElementById("blotter-head");
      columns.forEach(c => {{
        const th = document.createElement("th");
        th.textContent = c;
        head.appendChild(th);
      }});

      function render() {{
        const body = document.getElementById("blotter-body");
        body.innerHTML = "";
        const start = page * pageSize;
        const pageRows = filtered.slice(start, start + pageSize);
        pageRows.forEach(row => {{
          const tr = document.createElement("tr");
          columns.forEach(c => {{
            const td = document.createElement("td");
            const v = row[c];
            td.textContent = (v === null || v === undefined) ? "" : v;
            tr.appendChild(td);
          }});
          body.appendChild(tr);
        }});
        const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
        document.getElementById("blotter-page-info").textContent =
          `Page ${{page + 1}} of ${{totalPages}} (${{filtered.length}} rows)`;
      }}

      document.getElementById("blotter-search").addEventListener("input", (e) => {{
        const q = e.target.value.toLowerCase();
        filtered = allRows.filter(row => columns.some(c => String(row[c] ?? "").toLowerCase().includes(q)));
        page = 0;
        render();
      }});
      document.getElementById("blotter-prev").addEventListener("click", () => {{
        if (page > 0) {{ page -= 1; render(); }}
      }});
      document.getElementById("blotter-next").addEventListener("click", () => {{
        const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
        if (page < totalPages - 1) {{ page += 1; render(); }}
      }});

      render();
    }})();
    </script>
    """


_CSS = """
body { -webkit-font-smoothing: antialiased; font-family: -apple-system, Helvetica, Arial, sans-serif;
       font-size: 13px; line-height: 1.4; margin: 0; padding: 16px 20px; color: #111; background: #fff; }
.container { display: flex; align-items: flex-start; gap: 24px; width: 100%; }

h1 { font-weight: 400; margin: 0; font-size: 22px; }
h1 .date-range { display: inline; margin-left: 10px; font-size: 14px; font-weight: 400; color: #666; }
.byline { color: grey; margin: 4px 0 0; }
hr { margin: 20px 0 30px; height: 0; border: 0; border-top: 1px solid #ccc; }

#left { flex: 68 1 0; min-width: 0; }
#right { flex: 32 1 0; min-width: 0; }
#left h3, #right h3 { font-weight: 700; margin: 24px 0 10px; font-size: 14px; }
#left h3:first-child { margin-top: 0; }
.chart-block { margin-bottom: 8px; }

h2 { margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }

table.kpi-table { border-collapse: collapse; width: 100%; margin: 0 0 30px; border: 0; }
table.kpi-table th, table.kpi-table td { text-align: right; padding: 4px 5px 3px; border: 0; }
table.kpi-table td:first-of-type, table.kpi-table th:first-of-type { text-align: left; padding-left: 2px; }
table.kpi-table thead th { font-weight: 700; background: #eee; }

.full-width-section { margin: 40px 0 0; }
#blotter-section { margin-top: 12px; }
#blotter-search { width: 320px; padding: 6px; margin-right: 12px; }
table.blotter-table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
table.blotter-table th, table.blotter-table td { border: 1px solid #eee; padding: 4px 8px; text-align: right; }
table.blotter-table th:nth-child(-n+6), table.blotter-table td:nth-child(-n+6) { text-align: left; }

@media print {
  hr { margin: 20px 0; }
  body { padding: 0; }
  .container { display: block; }
  #left, #right { width: 100%; }
  #blotter-search, #blotter-prev, #blotter-next, #blotter-page-info { display: none; }
}
"""


def build_report(result_df: pd.DataFrame, portfolio: "Portfolio", trade_events: pd.DataFrame,
                  blotter: pd.DataFrame, config: dict, out_path: Path) -> Path:
    """Assemble all charts + KPI/drawdown tables + the trade blotter into one
    self-contained, portable HTML file at out_path (plotly.js is inlined, not
    CDN-linked, so the report is fully viewable offline)."""
    cum_with_friction = compute_cum_pnl(result_df)
    cum_without_friction = cum_pnl_without_friction(result_df)

    figs = [
        _pnl_chart_with_markers("Cumulative P&L", result_df, cum_without_friction, trade_events),
        _pnl_chart_with_markers("Cumulative P&L (with friction)", result_df, cum_with_friction, trade_events),
        _annual_bar_chart(result_df),
        _drawdown_chart(result_df),
        _daily_pnl_chart(result_df),
        _greek_chart(result_df, "delta", "Delta"),
        _greek_chart(result_df, "vega", "Vega"),
        _monthly_heatmap(result_df),
    ]
    plotly_config = {"responsive": True, "displaylogo": False}
    chart_divs = [
        _defer_chart_script(
            fig.to_html(full_html=False, include_plotlyjs=(i == 0), div_id=f"chart-{i}", config=plotly_config)
        )
        for i, fig in enumerate(figs)
    ]
    # Re-activate every chart's (now-inert) script once the whole page,
    # including the sidebar, has been parsed and laid out — see
    # _defer_chart_script.
    activate_charts_script = """
    <script>
    document.querySelectorAll('script.deferred-plotly').forEach(function (oldScript) {
      var newScript = document.createElement('script');
      newScript.textContent = oldScript.textContent;
      oldScript.parentNode.replaceChild(newScript, oldScript);
    });
    </script>
    """

    strategy_name = config.get("strategy", {}).get("name", "strategy")
    pair = config.get("pair", "")
    start_date, end_date = result_df["date"].iloc[0].date(), result_df["date"].iloc[-1].date()

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Backtester report — {strategy_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Backtester report: {strategy_name} ({pair}) <span class="date-range">{start_date} to {end_date}</span></h1>
  <p class="byline">Generated by fxbacktest</p>
  <hr>
  <div class="container">
    <div id="left">
      <h3>Performance</h3>
      <div class="chart-block">{chart_divs[0]}</div>
      <div class="chart-block">{chart_divs[1]}</div>
      <div class="chart-block">{chart_divs[2]}</div>
      <div class="chart-block">{chart_divs[3]}</div>
      <div class="chart-block">{chart_divs[4]}</div>
      <div class="chart-block">{chart_divs[5]}</div>
      <div class="chart-block">{chart_divs[6]}</div>
      <div class="chart-block">{chart_divs[7]}</div>
    </div>
    <div id="right">
      <h3>Key Performance Metrics</h3>
      {_kpi_table_html(result_df, trade_events, portfolio)}
      <h3>Worst 5 drawdowns</h3>
      {_worst_drawdowns_html(result_df)}
    </div>
  </div>

  <div class="full-width-section">
    <h2>Trade blotter</h2>
    {_blotter_section_html(blotter)}
  </div>

  {activate_charts_script}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
