"""Execution engine for strategy decisions."""

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
    """Coordinate plan generation and execution for a universe.

    The engine wires a :class:`~backtest_lib.engine.plan.PlanGenerator` and a
    :class:`~backtest_lib.engine.execute.PlanExecutor` to execute strategy
    decisions against a consistent security universe.
    """

    _planner: PlanGenerator[TPlanOp]
    _executor: PlanExecutor[TPlanOp]
    _universe: tuple[str, ...]

    def __init__(
        self,
        planner: PlanGenerator[TPlanOp],
        executor: PlanExecutor[TPlanOp],
        universe: Iterable[str],
    ):
        """Initialize the engine with planning and execution components.

        Args:
            planner: :class:`~backtest_lib.engine.plan.PlanGenerator` that
                translates decisions into execution plans.
            executor: :class:`~backtest_lib.engine.execute.PlanExecutor` that
                applies plans to the portfolio.
            universe: Securities considered tradable by the engine.

        """
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
        """Run a strategy once and execute its resulting plan.

        Args:
            strategy: :class:`~backtest_lib.strategy.Strategy` callable producing
                a decision.
            portfolio: :class:`~backtest_lib.portfolio.Portfolio` to update.
            market: :class:`~backtest_lib.market.MarketView` providing historical
                context.
            ctx: :class:`~backtest_lib.strategy.context.StrategyContext` passed
                to the strategy.
            prices: :class:`~backtest_lib.universe.universe_mapping.UniverseMapping`
                of current prices used for planning.

        Returns:
            :class:`~backtest_lib.engine.execute.ExecutionResult` after applying
            the plan.

        """
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
    """Create an engine from a planner and an executor.

    Args:
        planner: :class:`~backtest_lib.engine.plan.PlanGenerator` to use.
        executor: :class:`~backtest_lib.engine.execute.PlanExecutor` to use.
        universe: Securities considered tradable by the engine.
    """
    return Engine(planner, executor, universe)
