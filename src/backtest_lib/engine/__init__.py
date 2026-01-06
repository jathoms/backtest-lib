from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from backtest_lib.engine.execute import PlanExecutor
from backtest_lib.engine.plan import PlanGenerator
from backtest_lib.market import MarketView
from backtest_lib.strategy import Strategy
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe.universe_mapping import UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.engine.execute import ExecutionResult
    from backtest_lib.engine.plan import PlanOp
    from backtest_lib.portfolio import Portfolio


class Engine[TPlanOp: PlanOp]:
    _planner: PlanGenerator[TPlanOp]
    _executor: PlanExecutor[TPlanOp]
    _universe: tuple[str, ...]

    def __init__(
        self,
        planner: PlanGenerator[TPlanOp],
        executor: PlanExecutor[TPlanOp],
        universe: Iterable[str],
    ):
        self._planner = planner
        self._executor = executor
        self._universe = tuple(universe)

    def execute_strategy(
        self,
        strategy: Strategy,
        portfolio: Portfolio,
        market: MarketView,
        ctx: StrategyContext,
        prices: UniverseMapping,
    ) -> ExecutionResult:
        decision = strategy(
            universe=self._universe,
            current_portfolio=portfolio,
            market=market,
            ctx=ctx,
        )
        return self._executor.execute_plan(
            self._planner.generate_plan(decision, prices),
            portfolio=portfolio,
            prices=prices,
            market=market,
        )


def make_engine[TPlanOp: PlanOp](
    planner: PlanGenerator[TPlanOp],
    executor: PlanExecutor[TPlanOp],
    universe: Iterable[str],
) -> Engine[TPlanOp]:
    return Engine(planner, executor, universe)
