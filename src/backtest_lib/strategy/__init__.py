from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from backtest_lib.market import MarketView
from backtest_lib.universe import Universe
from backtest_lib.universe.universe_mapping import UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.strategy.context import StrategyContext

from backtest_lib.portfolio import WeightedPortfolio


@dataclass(frozen=True)
class Decision:
    target: WeightedPortfolio
    notes: UniverseMapping[Any] | None = None


@runtime_checkable
class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_portfolio: WeightedPortfolio,
        market: MarketView,
        ctx: StrategyContext | None,
    ) -> Decision: ...
