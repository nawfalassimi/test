from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    import pandas as pd

    from fxbacktest.execution.order import Order
    from fxbacktest.market.market import Market
    from fxbacktest.portfolio.portfolio import Portfolio


class Strategy(ABC):
    strategy_id: str
    required_pairs: List[str]

    @abstractmethod
    def generate_orders(self, date: "pd.Timestamp", market: "Market",
                        portfolio: "Portfolio") -> List["Order"]:
        ...


STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}


def register_strategy(name: str):
    def decorator(cls: Type[Strategy]) -> Type[Strategy]:
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> Type[Strategy]:
    return STRATEGY_REGISTRY[name]
