from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from fxbacktest.analytics.blotter import build_trade_blotter
from fxbacktest.analytics.kpis import calmar_ratio, sharpe_ratio, sortino_ratio
from fxbacktest.analytics.metrics import compute_cum_pnl, compute_drawdown
from fxbacktest.analytics.report import build_report
from fxbacktest.analytics.trades import extract_trade_events
from fxbacktest.data.schema import validate_quotes
from fxbacktest.data.synthetic import SyntheticFxDataGenerator, SyntheticVixGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.base import get_strategy

CONFIG_PATH = Path(__file__).parent / "fxbacktest" / "config" / "signal_strategy.yaml"


def load_quotes(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    dfs = [
        SyntheticFxDataGenerator(pair=pair, start=data_cfg["start"], end=data_cfg["end"],
                                 seed=data_cfg["seed"] + i).generate()
        for i, pair in enumerate(config["pairs"])
    ]
    df = pd.concat(dfs, ignore_index=True)
    validate_quotes(df)
    return df


def load_vix(config: dict) -> pd.DataFrame:
    vix_cfg = config["vix_data"]
    return SyntheticVixGenerator(start=vix_cfg["start"], end=vix_cfg["end"], seed=vix_cfg["seed"]).generate()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    quotes_df = load_quotes(config)
    vix_df = load_vix(config)

    pricer = GarmanKohlhagenPricer()
    strategy_cls = get_strategy(config["strategy"]["name"])
    strategy = strategy_cls(
        quotes_df=quotes_df, vix_df=vix_df, pairs=config["pairs"],
        tenor=config["strategy"]["tenor"], target_delta=config["strategy"]["target_delta"],
        notional=config["strategy"]["notional"],
    )
    hedger = DailyDeltaHedger(pricer, mode=config["hedging"]["mode"],
                              threshold=config["hedging"].get("threshold", 0.0))
    cost_model = TransactionCostModel.zero()

    result_df, portfolio = run_backtest(
        quotes_df, [strategy], hedger, pricer,
        assumed_foreign_rate=config["assumed_foreign_rate"],
        cost_model=cost_model, vix_df=vix_df,
    )

    cum_pnl = compute_cum_pnl(result_df)
    drawdown = compute_drawdown(result_df)
    max_dd = drawdown.min()
    max_dd_date = result_df["date"].iloc[drawdown.values.argmin()]

    print(f"Backtest: {result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()} ({len(result_df)} days)")
    print(f"Hedging mode:              {config['hedging']['mode']}")
    print(f"Total P&L (with friction): {cum_pnl.iloc[-1]:,.2f}")
    print(f"Total friction costs:      {result_df['friction_cost'].sum():,.2f}")
    print(f"Max drawdown:              {max_dd:,.2f} (on {max_dd_date.date()})")
    print(f"Sharpe / Sortino:          {sharpe_ratio(result_df):.2f} / {sortino_ratio(result_df):.2f}")
    print(f"Calmar (Annual P&L / MaxDD): {calmar_ratio(result_df):.2f}")

    strategy_positions = [p for p in portfolio.positions if p.strategy_id == strategy.strategy_id]
    early_closes = [p for p in strategy_positions if not p.is_open and p.exit_date < p.instrument.expiry]
    print(f"Entries: {len({p.clip_id for p in strategy_positions})}, early exits (hard-stop): {len({p.clip_id for p in early_closes})}")

    trade_events = extract_trade_events(portfolio)
    blotter = build_trade_blotter(portfolio, quotes_df, pricer, config["assumed_foreign_rate"])

    blotter_path = Path("output") / "signal_strategy_trade_blotter.xlsx"
    blotter_path.parent.mkdir(parents=True, exist_ok=True)
    blotter.to_excel(blotter_path, index=False)
    print(f"Saved trade blotter to {blotter_path} ({len(blotter)} rows)")

    report_path = build_report(result_df, portfolio, trade_events, blotter, config,
                               Path("output") / "signal_strategy_report.html")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
