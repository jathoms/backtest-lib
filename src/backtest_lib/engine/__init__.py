from __future__ import annotations

from typing import TYPE_CHECKING

from backtest_lib.engine.decision import Decision
from backtest_lib.engine.execute import PlanExecutor
from backtest_lib.engine.plan import PlanGenerator
from backtest_lib.market import MarketView
from backtest_lib.universe.universe_mapping import UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.engine.execute import ExecutionResult
    from backtest_lib.engine.plan import PlanOp
    from backtest_lib.portfolio import Portfolio


class Engine[TPlanOp: PlanOp]:
    _planner: PlanGenerator[TPlanOp]
    _executor: PlanExecutor[TPlanOp]

    def __init__(
        self, planner: PlanGenerator[TPlanOp], executor: PlanExecutor[TPlanOp]
    ):
        self._planner = planner
        self._executor = executor

    def execute_decision(
        self,
        decision: Decision,
        portfolio: Portfolio,
        prices: UniverseMapping,
        market: MarketView,
    ) -> ExecutionResult:
        return self._executor.execute_plan(
            self._planner.generate_plan(decision, prices),
            portfolio=portfolio,
            prices=prices,
            market=market,
        )


def make_engine[TPlanOp: PlanOp](
    planner: PlanGenerator[TPlanOp], executor: PlanExecutor[TPlanOp]
) -> Engine[TPlanOp]:
    return Engine(planner, executor)
