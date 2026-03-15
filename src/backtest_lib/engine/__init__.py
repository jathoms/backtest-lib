"""Execution engine for strategy decisions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
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

ALLOWED_STRATEGY_PARAMETERS = ("universe", "current_portfolio", "market", "ctx")


class Engine[TPlanOp: PlanOp]:
    """Coordinate plan generation and execution for a universe.

    The engine wires a :class:`~backtest_lib.engine.plan.PlanGenerator` and a
    :class:`~backtest_lib.engine.execute.PlanExecutor` to execute strategy
    decisions against a consistent security universe.
    """

    _planner: PlanGenerator[TPlanOp]
    _executor: PlanExecutor[TPlanOp]
    _universe: tuple[str, ...]
    _func_param_mapping: dict[Callable, tuple[str, ...]]

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
        self._func_param_mapping = {}

    def _parse_strategy_args(self, strategy: Strategy) -> tuple[str, ...]:
        if strategy in self._func_param_mapping:
            return self._func_param_mapping[strategy]
        args = tuple(inspect.signature(strategy).parameters.keys())
        unknown_args = [arg for arg in args if arg not in ALLOWED_STRATEGY_PARAMETERS]
        if unknown_args:
            raise TypeError(
                f"Invalid strategy parameters for '{strategy.__name__}': "
                f"{unknown_args}. The valid parameter names for a strategy are "
                f"{ALLOWED_STRATEGY_PARAMETERS}"
            )
        self._func_param_mapping[strategy] = args
        return args

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
        available_args = {
            "universe": self._universe,
            "current_portfolio": portfolio,
            "market": market,
            "ctx": ctx,
        }
        used_kwargs = {
            name: available_args[name]
            for name in self._parse_strategy_args(strategy)
            if name in available_args
        }
        decision = strategy(**used_kwargs)
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
