from __future__ import annotations

import argparse
from pathlib import Path

from fxbacktest.data.schema import validate_quotes
from fxbacktest.data.synthetic import SyntheticFxDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FX quote data for one pair.")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2023-12-29")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic/eurusd_quotes.csv")
    args = parser.parse_args()

    generator = SyntheticFxDataGenerator(pair=args.pair, start=args.start, end=args.end, seed=args.seed)
    df = generator.generate()
    validate_quotes(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()}) to {out_path}")


if __name__ == "__main__":
    main()
