from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import pandas as pd

from fxbacktest.data.schema import parse_pair
from fxbacktest.market.snapshot import MarketSnapshot


def bridge_pairs_for(traded_pairs: Set[str]) -> Set[str]:
    """For every traded pair with neither leg in USD (e.g. EURHUF), the
    {base}USD pair is needed to convert that pair's risk/P&L to USD (see
    Market.usd_rate's one-hop cross). Documented limitation: assumes the base
    currency is conventionally quoted as {base}USD (true for EUR, GBP, AUD...),
    not USD{ccy} (CHF, CAD...) — acceptable for now since real quote-convention
    handling belongs with the real data-provider work, not this mock-data milestone."""
    bridges = set()
    for pair in traded_pairs:
        base, quote = parse_pair(pair)
        if base != "USD" and quote != "USD":
            bridges.add(f"{base}USD")
    return bridges


@dataclass(frozen=True)
class Market:
    """A single date's market data across every currency pair currently loaded
    (one per traded pair, plus any USD-conversion bridge pairs). This is what
    lets the rest of the system express all risk/P&L in USD even when a
    traded pair doesn't involve USD at all (e.g. EURHUF)."""

    date: pd.Timestamp
    snapshots: Dict[str, MarketSnapshot]
    vix: Optional[float] = None

    def snapshot(self, pair: str) -> MarketSnapshot:
        return self.snapshots[pair]

    def usd_rate(self, currency: str) -> float:
        """USD per 1 unit of `currency`. Resolution order: (1) USD itself; (2)
        a directly loaded {currency}USD pair; (3) a directly loaded
        USD{currency} pair (inverted); (4) a one-hop cross through any loaded
        pair with `currency` as one leg and a USD-resolvable currency as the
        other. No further-than-one-hop search — the bridge-pair loading rule
        (see fxbacktest/engine/daily_loop.py) is designed to keep every
        currency actually in play reachable within one hop, so hitting this
        limit signals a genuine gap in the loaded pairs, not a missing feature."""
        if currency == "USD":
            return 1.0

        direct = f"{currency}USD"
        if direct in self.snapshots:
            return self.snapshots[direct].spot

        inverse = f"USD{currency}"
        if inverse in self.snapshots:
            return 1.0 / self.snapshots[inverse].spot

        for pair, snap in self.snapshots.items():
            base, quote = parse_pair(pair)
            if quote == currency and base != currency:
                other_rate = self._direct_usd_rate(base)
                if other_rate is not None:
                    return other_rate / snap.spot
            if base == currency and quote != currency:
                other_rate = self._direct_usd_rate(quote)
                if other_rate is not None:
                    return other_rate * snap.spot

        raise ValueError(
            f"cannot resolve a USD rate for {currency!r}: no direct or one-hop pair "
            f"loaded (loaded pairs: {sorted(self.snapshots)})"
        )

    def _direct_usd_rate(self, currency: str):
        """Steps (1)-(3) of usd_rate only (no cross-hop) — used internally to
        avoid recursing more than one hop deep."""
        if currency == "USD":
            return 1.0
        if f"{currency}USD" in self.snapshots:
            return self.snapshots[f"{currency}USD"].spot
        if f"USD{currency}" in self.snapshots:
            return 1.0 / self.snapshots[f"USD{currency}"].spot
        return None
