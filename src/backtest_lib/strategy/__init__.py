from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from backtest_lib.market import MarketView
from backtest_lib.universe import Universe, UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.strategy.context import StrategyContext

from backtest_lib.portfolio import WeightedPortfolio


@dataclass(frozen=True)
class Decision:
    target: WeightedPortfolio
    notes = UniverseMapping[Any] | None


class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_portfolio: WeightedPortfolio,
        market: MarketView,
        # schedule: Schedule, # for seeing where we are in the rebalance schedule
        ctx: StrategyContext | None,
    ) -> Decision: ...
