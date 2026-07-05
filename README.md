# FX Options + Delta-One Backtest System

> Project brief / README — paste this into a new conversation with Claude to kick
> off implementation. It captures the full scope, architecture, and design
> decisions already agreed on, so no context needs to be re-derived.

---

## Objective

Build a flexible, close-data-based backtesting system for FX option and delta-one
(spot / FX forward) strategies. The system must let me define strategies/signals
declaratively, price a multi-instrument FX portfolio daily, delta-hedge it, enforce
risk limits, and produce a full performance report — while staying easy to extend
with new strategies, signals, currencies, and data sources.

Assume Python (pandas, numpy, scipy, matplotlib/plotly) unless told otherwise.

## Available data

- **FX vol data**: ATM, Risk Reversal (RR), Butterfly (BF) quotes, at 10-delta and
  25-delta, across standard tenors, per currency pair. Sourced from Excel or an API.
- **Delta-one data**: spot and FX forward points per tenor, per currency pair.
  Sourced from Excel or an API.
- Data is **close-of-day only** — no intraday.

## Functional requirements

- Strategies/signals must support:
  - Calendar-based entry/exit (e.g. specific day of week/month)
  - Signal-based entry/exit (a computable quantitative signal, e.g. RR z-score, carry)
  - Multi-clip entry/exit (scaling in/out over several trades)
- Daily portfolio pricing (mark-to-market) to compute daily P&L
- Daily delta hedging — either every day, or only when a threshold is breached
- Risk limits: option size per currency, vega per currency, combined FX exposure
  (option + delta-one), configurable per currency
- The system must be easy to extend: new strategies, new signals, new currencies,
  new data sources, without touching core engine code
- Daily risk metrics on the portfolio: P&L, drawdown, FX exposure, delta, vega, gamma
- Final report with:
  - Plots: cumulative P&L, drawdown (underwater chart), yearly P&L and drawdown,
    daily P&L, daily delta, daily vega
  - Metrics: total P&L, max drawdown (+ date), Sharpe, Calmar, Sortino,
    best/worst 10-day P&L windows
  - A trade blotter (Excel) listing every trade/clip with entry/exit dates, size,
    signals, strategy, and exposure at entry (see "Output: trade blotter" below)

## Architecture — layered pipeline

```
Data sources (Excel/API)
        ↓
Market data layer (vol surface construction, FX forward curve)
        ↓
Pricing layer (FX option pricer, spot/forward pricer, Greeks)
        ↓
┌─────────────────────────────────────────────┐
│ Backtest engine — daily loop                 │
│  Signals & strategy   |   Risk limits         │
│  Hedging engine       |   Execution / clips    │
└─────────────────────────────────────────────┘
        ↓
Portfolio & positions (state: P&L, delta, vega, gamma, FX exposure)
        ↓
Analytics & reporting (metrics, plots)
```

Design principles:
1. **Layered** — each layer only talks to its neighbors through a clean interface.
2. **Config + registry driven** — strategies, signals, and risk limits are defined
   in YAML/config and dispatched to registered classes, not hardcoded branches.
3. **One immutable "market snapshot" per date** flows through the pipeline so every
   layer sees a consistent view of the world — this is the main defense against
   look-ahead bias.

## Layer-by-layer design

### 1. Data layer
- `MarketDataProvider` abstract interface, with `ExcelMarketDataProvider` and
  `APIMarketDataProvider` implementations.
- Validate on ingestion (missing tenors, stale quotes, inverted RR/BF signs).
- Cache to a normalized store (parquet/SQLite) — the engine never touches
  Excel/API directly during the daily loop.

### 2. Market layer (vol surface construction — the trickiest part)
- **Quotes → smile points** (5-point smile per tenor: 10P, 25P, ATM, 25C, 10C):
  ```
  vol(25C) ≈ vol_ATM + BF25 + RR25/2
  vol(25P) ≈ vol_ATM + BF25 - RR25/2
  (same pattern for 10-delta)
  ```
  (Note: this is the standard first-order approximation. BF as quoted is a market
  strangle, not literally `(vol_call+vol_put)/2 - vol_ATM`; a Vanna-Volga correction
  can be added later for exactness — not a v1 blocker.)
- **Interpolate across delta**: cubic spline (or SABR/SVI) through the 5 points.
- **Interpolate across tenor**: interpolate in **total variance** (`σ²·t`), not vol,
  to avoid calendar arbitrage, for expiries off the standard tenor grid.
- **Delta ↔ strike**: solve iteratively (fixed point) since vol depends on strike
  via the smile, and strike depends on vol via delta.
- `FXForwardCurve` derives implied carry `(rd - rf)` from spot/forward points, since
  we likely won't have clean separate discount curves from close data alone. Decide
  once whether to track undiscounted forward P&L or use a proxy discount curve for
  PV — must be consistent across pricer, hedge P&L, and reporting.

### 3. Instruments & pricing
- Instruments are dumb data containers: `FxVanillaOption`, `FxSpot`, `FxForward`.
- Pricers are separate (strategy pattern): `GarmanKohlhagenPricer` for options.
- Each pricer exposes `price(instrument, market_snapshot)` and
  `greeks(instrument, market_snapshot)` (delta, gamma, vega, theta).

### 4. Signals & strategy (registry pattern for extensibility)
- `Signal` abstract base — e.g. `DayOfWeekSignal`, `RRMeanReversionSignal`,
  `CarrySignal` — registered via a decorator (`@register_signal("name")`).
- `Strategy` abstract base — takes a config (universe, entry/exit rule, clip
  schedule, sizing) and generates orders each day. Registered the same way
  (`@register_strategy("name")`).
- New strategies/signals = new class + config file, never touching the engine.

### 5. Risk & hedging (two separate concerns)
- **Risk limits** = pre-trade gate: check proposed orders against max vega per
  currency, max option notional per currency, max net FX exposure; scale down or
  reject and log breaches.
- **Hedging engine** = standing process independent of strategies: recompute
  portfolio delta per currency daily; if mode is `daily`, flatten it; if
  `threshold`, only trade when `|delta| > limit`.

### 6. Backtest engine (daily loop — orchestration only, no business logic)
```
for date in trading_calendar:
    market = build_market_snapshot(date)
    portfolio.mark_to_market(market)
    daily_metrics.record(date, portfolio.snapshot())

    hedge_orders = hedging_engine.rehedge_orders(portfolio, market)
    strategy_orders = [s.generate_orders(date, market, portfolio, signals)
                        for s in active_strategies]

    approved = risk_engine.check(hedge_orders + strategy_orders, portfolio, limits)
    portfolio.execute(approved, market, date)
```
- Decide explicitly and document: same-day-close vs next-close execution timing
  (with close-only data, this is a deliberate parameter, not an accident).

### 7. Portfolio & positions
- `Position` tagged with `clip_id` and `strategy_id` from day one (cheap to add,
  enables later P&L attribution by strategy/clip).
- `Portfolio` aggregates: `pnl_by_ccy`, `delta_by_ccy`, `vega_by_ccy`,
  `net_fx_exposure`, gamma.

### 8. Analytics & reporting (asset-class-agnostic — reusable later)
- Input: daily metrics DataFrame (date, pnl, cum_pnl, delta/vega/gamma by ccy,
  fx_exposure).
- Metrics: cumulative P&L, drawdown series + max DD + its date + duration, yearly
  P&L/DD table, Sharpe, Sortino, Calmar, best/worst 10-day rolling P&L.
- Plots: cumulative P&L, underwater drawdown chart, daily P&L bars, daily
  delta/vega lines, yearly P&L/DD bars.

## Suggested repo structure

```
fxbacktest/
  data/            providers, cache, validation
  market/          vol surface, fwd curve, market snapshot
  instruments/     option, spot, forward definitions
  pricing/         pricers, greeks
  portfolio/       position, portfolio, ledger
  signals/         signal registry
  strategies/      strategy registry
  risk/            risk limits engine
  hedging/         hedging engine
  execution/       order, clip logic
  engine/          the daily-loop orchestrator
  analytics/       performance metrics, plots, report generation
  config/          strategy/risk YAML files
  tests/           unit tests per layer
```

## Gotchas to preserve across the build

- Interpolate variance (not vol) across tenor to avoid calendar arbitrage.
- Make transaction costs (bid/ask spread on vol and spot) a config parameter from
  the start, not an afterthought.
- Pick one discounting convention and apply it consistently everywhere.
- Pick one execution timing convention (same-close vs next-close) and document it.
- Sanity-check the pricer + hedger together: daily hedged P&L on an ATM straddle
  should roughly track `0.5·gamma·(dS)² − theta·dt` — if that identity doesn't
  hold, something's wrong before strategy signals even enter the picture.

## Output: trade blotter (Excel)

Alongside the performance report, produce a trade blotter as an Excel file — one
row per trade/clip — with at least:

- Entry date, exit date
- Currency pair, instrument type (option/spot/forward), strike, tenor/expiry
- Size / notional, clip id
- Strategy id, signal value(s) that triggered entry/exit
- Entry price/vol, exit price/vol
- Realized P&L, and exposure at entry (delta, vega, FX notional)

This gives a human-auditable record of every position the backtest took, and is
usually the fastest way to sanity-check that the strategy logic is doing what you
think it's doing.

## Suggested first milestone

Scaffold the project: folder structure, a working `GarmanKohlhagenPricer`, the
vol surface builder (steps above), a `MarketSnapshot` object, and one toy
strategy (e.g. calendar-based short-vol carry) running through the daily loop
end to end on a single currency pair — before generalizing to the full
signal/strategy/risk registry and multi-currency support.
