from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import assert_never

import numpy as np

from backtest_lib.engine.execute import NO_COST, ExecutionResult, Trades
from backtest_lib.engine.plan import (
    MakeTradeOp,
    Plan,
    TargetHoldingsOp,
    TargettingOp,
    TargetWeightsOp,
)
from backtest_lib.engine.plan.perfect_world import PerfectWorldOps
from backtest_lib.market import MarketView, get_mapping_type_from_backend
from backtest_lib.portfolio import (
    Portfolio,
    QuantityPortfolio,
    WeightedPortfolio,
    make_universe_mapping,
)
from backtest_lib.universe.universe_mapping import UniverseMapping


class TargettingOpNotFirstException(Exception): ...


class DuplicateTargetException(Exception): ...


class NegativeCashException(Exception): ...


logger = logging.getLogger(__name__)


def _warn_redundant_cash(amount: float) -> None:
    warnings.warn(
        "fill_cash is set to True, but you have "
        "manually specified an amount of cash in this "
        f"TargetWeightsDecision. The passed value of {amount} "
        "will be ignored.",
        stacklevel=2,
    )


class PerfectWorldPlanExecutor:
    _backend: str
    _security_alignment: tuple[str, ...]

    def __init__(self, backend: str, security_alignment: Iterable[str]):
        self._backend = backend
        self._security_alignment = tuple(security_alignment)

    @cached_property
    def _backend_mapping_type(self) -> type[UniverseMapping]:
        return get_mapping_type_from_backend(self._backend)

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
            if isinstance(op, TargetWeightsOp):
                cash = op.cash or 0.0
                if op.fill_cash:
                    if op.cash is not None:
                        _warn_redundant_cash(op.cash)
                    cash = 1.0 - (
                        op.weights.sum()
                        if isinstance(op.weights, UniverseMapping)
                        else sum(op.weights.values())
                    )
                target_weights_universe_mapping = make_universe_mapping(
                    op.weights, self._security_alignment, self._backend
                )
                new_total_weight = target_weights_universe_mapping.sum() + cash
                assert np.isclose(new_total_weight, 1.0), (
                    "Total weight after making a decision must be 1.0, "
                    f"weight was {new_total_weight}. "
                    f"To automatically calculate cash as missing weights, "
                    "use `fill_cash=True` in `target_weights`"
                )

                portfolio_after = WeightedPortfolio(
                    universe=self._security_alignment,
                    holdings=target_weights_universe_mapping,
                    cash=cash,
                    total_value=portfolio.total_value,
                    constructor_backend=self._backend,
                )
            elif isinstance(op, TargetHoldingsOp):
                target_holdings_universe_mapping = make_universe_mapping(
                    op.holdings, self._security_alignment, self._backend
                )
                cash = op.cash or 0.0
                if op.fill_cash:
                    if op.cash is not None:
                        _warn_redundant_cash(op.cash)
                    target_asset_values = op.holdings * prices
                    target_total_value = target_asset_values.sum()
                    logger.debug(
                        f"asset values: {target_asset_values}, "
                        f"total: {target_total_value}"
                    )
                    cash = portfolio.total_value - target_total_value

                new_total_value = (
                    target_holdings_universe_mapping * prices
                ).sum() + cash

                assert np.isclose(new_total_value, portfolio.total_value), (
                    "Total holdings after making a decision must "
                    "preserve total value of holdings, including cash "
                    f"value changed from {portfolio.total_value} to {new_total_value}."
                    "To automatically calculate cash as missing holdings, "
                    "use `fill_cash=True` in `target_holdings`"
                )

                portfolio_after = WeightedPortfolio(
                    universe=self._security_alignment,
                    holdings=target_holdings_universe_mapping,
                    cash=cash,
                    total_value=portfolio.total_value,
                    constructor_backend=self._backend,
                )
            elif isinstance(op, Trades):
                # INVARIANT: the security alignment of the
                # Trades is the same as the Portfolio.
                # TODO: add a real check for the invariant
                qtys = portfolio.into_quantities(prices=prices)
                pos_delta = op.position_delta

                decision_cost = op.total_cost()
                new_cash = qtys.cash - decision_cost
                logger.debug(f"we had {qtys.cash}, now {new_cash}")

                if new_cash < 0:
                    # TODO: add settings for when this raises vs gives a warning etc.
                    raise NegativeCashException()

                new_holdings = qtys.holdings + pos_delta

                # TODO: This implicitly converts the user's portfolio into a
                # QuantityPortfolio when they make explicit trades. Review
                portfolio_after = QuantityPortfolio(
                    holdings=new_holdings,
                    cash=new_cash,
                    total_value=portfolio.total_value,
                    constructor_backend=self._backend,
                    universe=self._security_alignment,
                )
            else:
                assert_never(op)
        return ExecutionResult(
            before=portfolio, after=portfolio_after, costs=NO_COST, fills=None
        )
