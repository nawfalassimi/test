from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fxbacktest.analytics.metrics import compute_cum_pnl, compute_drawdown
from fxbacktest.data.schema import validate_quotes
from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.engine.daily_loop import run_backtest
from fxbacktest.execution.transaction_costs import TransactionCostModel
from fxbacktest.hedging.delta_hedger import DailyDeltaHedger
from fxbacktest.pricing.garman_kohlhagen import GarmanKohlhagenPricer
from fxbacktest.strategies.base import get_strategy

CONFIG_PATH = Path(__file__).parent / "fxbacktest" / "config" / "milestone1.yaml"


def load_quotes(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    path = Path(data_cfg["path"])
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    generator = SyntheticFxDataGenerator(
        pair=config["pair"], start=data_cfg["start"], end=data_cfg["end"], seed=data_cfg["seed"],
    )
    df = generator.generate()
    validate_quotes(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    quotes_df = load_quotes(config)

    pricer = GarmanKohlhagenPricer()
    strategy_cls = get_strategy(config["strategy"]["name"])
    strategy = strategy_cls(
        pair=config["pair"], tenor_days=config["strategy"]["tenor_days"],
        entry_weekday=config["strategy"]["entry_weekday"], notional=config["strategy"]["notional"],
    )
    hedger = DailyDeltaHedger(pricer, mode=config["hedging"]["mode"])
    cost_model = TransactionCostModel.from_config(config["transaction_costs"])

    result = run_backtest(
        quotes_df, strategy, hedger, pricer,
        assumed_foreign_rate=config["assumed_foreign_rate"],
        cost_model=cost_model,
    )

    cum_pnl = compute_cum_pnl(result)
    drawdown = compute_drawdown(result)
    max_dd = drawdown.min()
    max_dd_date = result["date"].iloc[drawdown.values.argmin()]

    print(f"Backtest: {result['date'].iloc[0].date()} to {result['date'].iloc[-1].date()} ({len(result)} days)")
    print(f"Total P&L:      {cum_pnl.iloc[-1]:,.2f}")
    print(f"Max drawdown:   {max_dd:,.2f} (on {max_dd_date.date()})")
    print(f"Final cum P&L:  {result['cum_pnl'].iloc[-1]:,.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["date"], cum_pnl)
    ax.set_title("Cumulative P&L — short_vol_carry_1m")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    fig.tight_layout()

    out_path = Path("output") / "cum_pnl.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved cumulative P&L plot to {out_path}")


if __name__ == "__main__":
    main()
