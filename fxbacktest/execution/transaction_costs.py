from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

CostKind = Literal["vol_spread", "premium_spread"]


@dataclass(frozen=True)
class OptionCostSpec:
    kind: CostKind = "vol_spread"
    vol_spread_bp: float = 0.0       # half-spread in vol points; used when kind == "vol_spread"
    premium_spread_bp: float = 0.0   # half-spread as bp of premium; used when kind == "premium_spread"


@dataclass(frozen=True)
class SpotCostSpec:
    spread_pips: float = 0.0


@dataclass(frozen=True)
class PairCostSpec:
    option: OptionCostSpec = field(default_factory=OptionCostSpec)
    spot: SpotCostSpec = field(default_factory=SpotCostSpec)


class TransactionCostModel:
    """Per-currency-pair, per-product transaction costs. A pair not listed in
    `by_pair` falls back to `default`. Options can be costed either as a vol
    bid/ask spread (vega-based price impact) or a premium bid/ask spread
    (a spread on the premium itself) — set per pair via OptionCostSpec.kind.
    """

    def __init__(self, by_pair: Optional[Dict[str, PairCostSpec]] = None,
                 default: Optional[PairCostSpec] = None):
        self.by_pair = by_pair or {}
        self.default = default or PairCostSpec()

    def for_pair(self, pair: str) -> PairCostSpec:
        return self.by_pair.get(pair, self.default)

    def option_cost(self, pair: str, side: str, fair_price: float, vega: float) -> float:
        spec = self.for_pair(pair).option
        cost_sign = 1.0 if side == "buy" else -1.0
        if spec.kind == "vol_spread":
            return cost_sign * 0.5 * (spec.vol_spread_bp / 10_000.0) * vega
        if spec.kind == "premium_spread":
            return cost_sign * 0.5 * (spec.premium_spread_bp / 10_000.0) * fair_price
        raise ValueError(f"unknown option cost kind: {spec.kind}")

    def spot_cost(self, pair: str, side: str, notional: float) -> float:
        spec = self.for_pair(pair).spot
        cost_sign = 1.0 if side == "buy" else -1.0
        return cost_sign * 0.5 * (spec.spread_pips / 10_000.0) * notional

    @classmethod
    def zero(cls) -> "TransactionCostModel":
        return cls()

    @classmethod
    def from_config(cls, config: Dict[str, dict]) -> "TransactionCostModel":
        def _pair_spec(spec: dict) -> PairCostSpec:
            opt = spec.get("option", {})
            spot = spec.get("spot", {})
            return PairCostSpec(
                option=OptionCostSpec(
                    kind=opt.get("kind", "vol_spread"),
                    vol_spread_bp=opt.get("vol_spread_bp", 0.0),
                    premium_spread_bp=opt.get("premium_spread_bp", 0.0),
                ),
                spot=SpotCostSpec(spread_pips=spot.get("spread_pips", 0.0)),
            )

        by_pair = {pair: _pair_spec(spec) for pair, spec in config.items() if pair != "default"}
        default = _pair_spec(config["default"]) if "default" in config else PairCostSpec()
        return cls(by_pair=by_pair, default=default)
