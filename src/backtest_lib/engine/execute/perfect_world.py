from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import assert_never

import numpy as np

from backtest_lib.engine.decision import ReallocationMode
from backtest_lib.engine.execute import NO_COST, ExecutionResult, Trades
from backtest_lib.engine.plan import (
    MakeTradeOp,
    Plan,
    ReallocateOp,
    TargetHoldingsOp,
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


@dataclass(slots=True)
class _HoldingsReallocation:
    inner: UniverseMapping[int]


@dataclass(slots=True)
class _WeightsReallocation:
    inner: UniverseMapping[float]


_CompiledReallocation = _HoldingsReallocation | _WeightsReallocation


def _warn_redundant_cash(amount: float) -> None:
    warnings.warn(
        "fill_cash is set to True, but you have "
        "manually specified an amount of cash in this "
        f"TargetWeightsDecision. The passed value of {amount} "
        "will be ignored.",
        stacklevel=2,
    )


def _fill_cash_holdings(
    holdings: UniverseMapping[int],
    pf_total_value: float,
    prices: UniverseMapping[float],
) -> float:
    target_asset_values = holdings * prices
    target_total_value = target_asset_values.sum()
    logger.debug(f"asset values: {target_asset_values}, total: {target_total_value}")
    return pf_total_value - target_total_value


@dataclass(slots=True)
class _TargetWeightsCompiledOp:
    weights: UniverseMapping[float]
    cash: float

    def reallocate(self, reallocation: _WeightsReallocation) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Applying weights reallocation: {reallocation.inner}\n"
                f"to target weights {self.weights}"
            )
        assert np.isclose(reallocation.inner.sum(), 0)
        self.weights = self.weights + reallocation.inner


@dataclass(slots=True)
class _TargetHoldingsCompiledOp:
    holdings: UniverseMapping[int]
    cash: float

    def reallocate(self, reallocation: _HoldingsReallocation) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Applying holdings reallocation: {reallocation.inner}\n"
                f"to target holdings {self.holdings}"
            )
        assert np.isclose(reallocation.inner.sum(), 0)
        self.holdings = self.holdings + reallocation.inner


_CompiledOps = _TargetWeightsCompiledOp | _TargetHoldingsCompiledOp | Trades


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
        self,
        atomic_ops: Iterable[PerfectWorldOps],
        portfolio: Portfolio,
        prices: UniverseMapping,
    ) -> Iterator[_CompiledOps]:
        """Always yields the targetting op first"""
        trades = []
        compiled_targetting_op: (
            _TargetHoldingsCompiledOp | _TargetWeightsCompiledOp | None
        ) = None
        compiled_reallocation: _CompiledReallocation | None = None

        for op in atomic_ops:
            if isinstance(op, MakeTradeOp):
                trades.append(op.trade)
            elif isinstance(op, TargetWeightsOp):
                if compiled_targetting_op is not None:
                    raise DuplicateTargetException()
                target_weights_universe_mapping = make_universe_mapping(
                    op.weights,
                    universe=self._security_alignment,
                    constructor_backend=self._backend,
                )
                cash = op.cash or 0.0
                if op.fill_cash:
                    if op.cash is not None:
                        _warn_redundant_cash(op.cash)
                    cash = 1.0 - target_weights_universe_mapping.sum()
                new_total_weight = target_weights_universe_mapping.sum() + cash
                assert np.isclose(new_total_weight, 1.0), (
                    "Total weight after making a decision must be 1.0, "
                    f"weight was {new_total_weight}. "
                    f"To automatically calculate cash as missing weights, "
                    "use `fill_cash=True` in `target_weights`"
                )
                compiled_targetting_op = _TargetWeightsCompiledOp(
                    target_weights_universe_mapping,
                    cash=1.0 - target_weights_universe_mapping.sum(),
                )
            elif isinstance(op, TargetHoldingsOp):
                if compiled_targetting_op is not None:
                    raise DuplicateTargetException()
                target_quantities_universe_mapping = make_universe_mapping(
                    op.holdings,
                    universe=self._security_alignment,
                    constructor_backend=self._backend,
                )
                cash = op.cash or 0.0
                if op.fill_cash:
                    if op.cash is not None:
                        _warn_redundant_cash(op.cash)
                    cash = _fill_cash_holdings(
                        target_quantities_universe_mapping,
                        pf_total_value=portfolio.total_value,
                        prices=prices,
                    )
                new_total_value = (
                    target_quantities_universe_mapping * prices
                ).sum() + cash
                assert np.isclose(new_total_value, portfolio.total_value), (
                    "Total holdings after making a decision must "
                    "preserve total value of holdings, including cash "
                    f"value changed from {portfolio.total_value} to {new_total_value}."
                    "To automatically calculate cash as missing holdings, "
                    "use `fill_cash=True` in `target_holdings`"
                )
                compiled_targetting_op = _TargetHoldingsCompiledOp(
                    target_quantities_universe_mapping,
                    cash=_fill_cash_holdings(
                        target_quantities_universe_mapping,
                        pf_total_value=portfolio.total_value,
                        prices=prices,
                    ),
                )
            elif isinstance(op, ReallocateOp):
                fraction = op.inner.fraction
                in_wt = fraction / len(op.inner.to_securities)
                to_reallocation = make_universe_mapping(
                    {sec: in_wt for sec in op.inner.to_securities},
                    universe=self._security_alignment,
                    constructor_backend=self._backend,
                )
                if op.inner.mode is ReallocationMode.EQUAL_OUT_EQUAL_IN:
                    out_wt = fraction / len(op.inner.from_securities)
                    from_reallocation = make_universe_mapping(
                        {sec: -out_wt for sec in op.inner.from_securities},
                        universe=self._security_alignment,
                        constructor_backend=self._backend,
                    )
                    compiled_reallocation = _WeightsReallocation(
                        from_reallocation + to_reallocation
                    )
                elif op.inner.mode is ReallocationMode.PRO_RATA_OUT_EQUAL_IN:
                    holdings_values = portfolio.holdings * prices
                    total_from_value = sum(
                        holdings_values[sec] for sec in op.inner.from_securities
                    )

                    if total_from_value <= 0:
                        # TODO: add setting for when this silently scales down buys,
                        # warns, or errors
                        raise ValueError(
                            "The securities in 'out_of' are not held in the portfolio."
                            " Cannot perform a reallocation out of them."
                        )

                    from_reallocation = make_universe_mapping(
                        {
                            sec: (-holdings_values[sec] / total_from_value) * fraction
                            for sec in op.inner.from_securities
                        },
                        universe=self._security_alignment,
                        constructor_backend=self._backend,
                    )
                    compiled_reallocation = _WeightsReallocation(
                        from_reallocation + to_reallocation
                    )
                else:
                    assert_never(op.inner.mode)
            else:
                assert_never(op)

        logger.debug(
            "Finished parsing reallocation and target ops, combining "
            f"type(reallocation) : {type(compiled_reallocation)} with "
            f"type(target) : {type(compiled_targetting_op)}"
        )

        # if no target is provided, set the current portfolio as a base for the
        # reallocation
        if compiled_targetting_op is None and compiled_reallocation is not None:
            if isinstance(portfolio, WeightedPortfolio):
                compiled_targetting_op = _TargetWeightsCompiledOp(
                    portfolio.holdings, portfolio.cash
                )
            elif isinstance(portfolio, QuantityPortfolio):
                compiled_targetting_op = _TargetHoldingsCompiledOp(
                    portfolio.holdings, portfolio.cash
                )
            else:
                raise NotImplementedError(
                    f"Unsupported portfolio type: {type(portfolio)}"
                )

        if isinstance(compiled_targetting_op, _TargetWeightsCompiledOp):
            if isinstance(compiled_reallocation, _WeightsReallocation):
                compiled_targetting_op.reallocate(compiled_reallocation)
            elif isinstance(compiled_reallocation, _HoldingsReallocation):
                holdings_changes = compiled_reallocation.inner
                value_changes = holdings_changes * prices
                weight_changes = value_changes / portfolio.total_value
                compiled_targetting_op.reallocate(_WeightsReallocation(weight_changes))
            elif compiled_reallocation is not None:
                assert_never(compiled_reallocation)
            yield compiled_targetting_op
        elif isinstance(compiled_targetting_op, _TargetHoldingsCompiledOp):
            if isinstance(compiled_reallocation, _HoldingsReallocation):
                compiled_targetting_op.reallocate(compiled_reallocation)
            elif isinstance(compiled_reallocation, _WeightsReallocation):
                weight_changes = compiled_reallocation.inner
                target_value_changes = weight_changes * portfolio.total_value
                qty_changes = (target_value_changes / prices).truncate()
                actual_value_changes = qty_changes * prices
                cash_delta = -(actual_value_changes.sum())
                compiled_targetting_op.reallocate(_HoldingsReallocation(qty_changes))
                compiled_targetting_op.cash += cash_delta
            elif compiled_reallocation is not None:
                assert_never(compiled_reallocation)
            yield compiled_targetting_op
        elif compiled_targetting_op is not None:
            assert_never(compiled_targetting_op)

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
        Any targetting op will instantly change the portfolio to that target, so the
        convention in ``self._normalize_ops`` is to yield the target first.
        """
        del market

        portfolio_after = portfolio
        for op in self._normalize_ops(plan.steps, portfolio, prices):
            logger.debug(f"Executing compiled op {op}")
            if isinstance(op, _TargetWeightsCompiledOp):
                portfolio_after = WeightedPortfolio(
                    universe=self._security_alignment,
                    holdings=op.weights,
                    cash=op.cash,
                    total_value=portfolio.total_value,
                    constructor_backend=self._backend,
                )
            elif isinstance(op, _TargetHoldingsCompiledOp):
                portfolio_after = QuantityPortfolio(
                    universe=self._security_alignment,
                    holdings=op.holdings,
                    cash=op.cash,
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
