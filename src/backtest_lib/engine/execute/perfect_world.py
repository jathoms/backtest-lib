from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import assert_never

from backtest_lib.engine.execute import NO_COST, ExecutionResult, Trades
from backtest_lib.engine.plan import MakeTradeOp, Plan, TargettingOp
from backtest_lib.engine.plan.perfect_world import PerfectWorldOps
from backtest_lib.market import MarketView, get_mapping_type_from_mapping
from backtest_lib.portfolio import Portfolio, QuantityPortfolio
from backtest_lib.universe.universe_mapping import UniverseMapping


class TargettingOpNotFirstException(Exception): ...


class DuplicateTargetException(Exception): ...


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

    def _normalize_ops(
        self, atomic_ops: Iterable[PerfectWorldOps]
    ) -> Iterator[TargettingOp | Trades]:
        trades = []
        targetting_op: TargettingOp | None = None

        for op in atomic_ops:
            if isinstance(op, MakeTradeOp):
                trades.append(op.trade)
            elif isinstance(op, TargettingOp):
                if targetting_op is not None:
                    raise DuplicateTargetException()
                targetting_op = op
                yield op
            else:
                assert_never(op)
        if trades:
            yield Trades(
                trades=tuple(trades),
                security_alignment=self._security_alignment,
                backend_mapping_type=self._backend_mapping_type,
            )

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

        portfolio_after = portfolio
        for op in self._normalize_ops(plan.steps):
            if isinstance(op, TargettingOp):
                portfolio_after = op.to_portfolio(
                    total_value=portfolio.total_value, backend=self._backend
                )
            elif isinstance(op, Trades):
                if op.security_alignment != self._security_alignment:
                    raise ValueError(
                        "Passed op with misaligned securities into executor."
                    )
                pos_delta = op.position_delta
                decision_cost = op.total_cost()
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
                    total_value=portfolio.total_value,
                    constructor_backend=self._backend,
                )
            else:
                assert_never(op)
        return ExecutionResult(
            before=portfolio, after=portfolio_after, costs=NO_COST, fills=None
        )
