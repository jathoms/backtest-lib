"""Plan representations for executing decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from backtest_lib.engine.decision import Decision, ReallocateDecision, TradeDirection
from backtest_lib.universe.universe_mapping import UniverseMapping


@dataclass(frozen=True)
class TradeOrder:
    """Order describing a trade for a single security.

    Trade orders are produced by plan generators and executed by a
    :class:`~backtest_lib.engine.execute.PlanExecutor`.
    ``direction`` and ``qty`` together define the signed position change, while
    ``price`` is the execution price used to compute cash impact.
    """

    direction: TradeDirection
    qty: int
    security: str
    price: float

    @property
    def signed_qty(self) -> int:
        """Return signed quantity based on trade direction."""
        return -self.qty if self.direction == "sell" else self.qty

    def cost(self) -> float:
        """Return the cash impact of the trade."""
        return self.signed_qty * self.price


class PlanOp:
    """Marker base class for plan operations.

    Plan operations are the atomic instructions inside a
    :class:`~backtest_lib.engine.plan.Plan` that a
    :class:`~backtest_lib.engine.execute.PlanExecutor` can apply.
    """

    ...


TPlanOp_co = TypeVar("TPlanOp_co", bound=PlanOp, infer_variance=True)
T_co = TypeVar("T_co", covariant=True)


class PlanGenerator[TPlanOp_co](Protocol):
    """Protocol for converting decisions into executable plans.

    Implementations translate :data:`~backtest_lib.engine.decision.Decision`
    instances into a :class:`~backtest_lib.engine.plan.Plan` for execution.
    """

    def generate_plan(
        self, decision: Decision, prices: UniverseMapping
    ) -> Plan[TPlanOp_co]:
        """Generate a plan for a decision using current prices."""
        ...


@dataclass(frozen=True, slots=True)
class TargetWeightsOp(PlanOp):
    """Plan operation targeting portfolio weights.

    Typically produced from a
    :func:`~backtest_lib.engine.decision.target_weights` decision.
    """

    weights: Mapping[str, float]
    cash: float | None
    fill_cash: bool


@dataclass(frozen=True, slots=True)
class TargetHoldingsOp(PlanOp):
    """Plan operation targeting discrete holdings.

    Typically produced from a
    :func:`~backtest_lib.engine.decision.target_holdings` decision.
    """

    holdings: Mapping[str, int]
    cash: float | None
    fill_cash: bool


@dataclass(frozen=True, slots=True)
class MakeTradeOp(PlanOp):
    """Plan operation for executing a trade.

    Typically produced from a
    :func:`~backtest_lib.engine.decision.trade` decision.
    """

    trade: TradeOrder


@dataclass(frozen=True, slots=True)
class ReallocateOp(PlanOp):
    """Plan operation for reallocating between security sets.

    Typically produced from a
    :func:`~backtest_lib.engine.decision.reallocate` decision.
    """

    inner: ReallocateDecision


@dataclass(frozen=True, slots=True)
class Plan[TPlanOp_co]:
    """Ordered collection of plan operations.

    Plans are the input language for a
    :class:`~backtest_lib.engine.execute.PlanExecutor`.
    """

    steps: tuple[TPlanOp_co, ...]


TargettingOp = TargetWeightsOp | TargetHoldingsOp
