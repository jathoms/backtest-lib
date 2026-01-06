from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import assert_never

from backtest_lib.engine.execute import NO_COST, ExecutionResult
from backtest_lib.engine.plan import MakeTradesOp, Plan, TargettingOp
from backtest_lib.engine.plan.perfect_world import PerfectWorldOps
from backtest_lib.market import MarketView, get_mapping_type_from_mapping
from backtest_lib.portfolio import Portfolio, QuantityPortfolio
from backtest_lib.universe.universe_mapping import UniverseMapping


class TargettingOpNotFirstException(Exception): ...


class NegativeCashException(Exception): ...


class PerfectWorldPlanExecutor:
    _backend: str
    _security_alignment: tuple[str, ...]

    def __init__(self, backend: str, security_alignment: Iterable[str]):
        self._backend = backend
        self._security_alignment = tuple(security_alignment)

    @cached_property
    def _backend_mapping_type(self) -> type[UniverseMapping]:
        return get_mapping_type_from_mapping(self._backend)

    def execute_plan(
        self,
        plan: Plan[PerfectWorldOps],
        portfolio: Portfolio,
        prices: UniverseMapping,
        market: MarketView,
    ) -> ExecutionResult:
        """
        Executes a plan with assumptions of perfect conditions i.e no fees or slippage.
        The convention for the input plan for the PerfectWorldPlanExecutor is, if there
        is a TargettingOp as part of the plan, it must be the first operation in the
        plan, or else we throw an exception.

        """
        del market
        if len(plan) == 0:
            return ExecutionResult(
                before=portfolio,
                after=portfolio,
                costs=NO_COST,
                fills=None,
            )

        portfolio_after = portfolio
        have_target = False
        if isinstance(plan.steps[0], TargettingOp):
            portfolio_after = plan.steps[0].to_portfolio(
                total_value=portfolio.total_value, backend=self._backend
            )
            have_target = True

        for op in plan.steps[1:] if have_target else plan.steps:
            if isinstance(op, TargettingOp):
                raise TargettingOpNotFirstException()
            elif isinstance(op, MakeTradesOp):
                if op.trades.security_alignment != self._security_alignment:
                    raise ValueError(
                        "Passed op with misaligned securities into executor."
                    )
                pos_delta = op.trades.position_delta
                decision_cost = op.trades.total_cost()
                new_cash = portfolio.cash - decision_cost

                if new_cash < 0:
                    # TODO: add settings for when this raises vs gives a warning etc.
                    raise NegativeCashException()

                qtys = portfolio.into_quantities(prices=prices)
                new_holdings = qtys.holdings + pos_delta

                # TODO: This implicitly converts the user's portfolio into a
                # QuantityPortfolio when they make explicit trades. Review
                portfolio_after = QuantityPortfolio(
                    holdings=new_holdings,
                    cash=new_cash,
                    total_value=portfolio_after.total_value,
                    constructor_backend=self._backend,
                )
            else:
                assert_never(op)
        return ExecutionResult(
            before=portfolio, after=portfolio_after, costs=NO_COST, fills=None
        )
