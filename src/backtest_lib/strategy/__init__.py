from typing import Any, Protocol

from backtest_lib.market import MarketView
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Universe, UniverseMapping


class Holdings: ...


class Decision:
    target: Holdings
    notes = UniverseMapping[Any] | None


class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_holdings: Holdings,
        market: MarketView,
        # schedule: Schedule, # for seeing where we are in the rebalance schedule
        ctx: StrategyContext,
    ) -> Decision: ...
