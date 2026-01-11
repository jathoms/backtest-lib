from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from backtest_lib.engine.decision import Decision, ReallocateDecision, TradeDirection
from backtest_lib.universe.universe_mapping import UniverseMapping


@dataclass(frozen=True)
class TradeOrder:
    direction: TradeDirection
    qty: int
    security: str
    price: float

    @property
    def signed_qty(self) -> int:
        return -self.qty if self.direction == "sell" else self.qty

    def cost(self) -> float:
        return self.signed_qty * self.price


class PlanOp: ...


TPlanOp_co = TypeVar("TPlanOp_co", bound=PlanOp, infer_variance=True)
T_co = TypeVar("T_co", covariant=True)


class PlanGenerator[TPlanOp_co](Protocol):
    def generate_plan(
        self, decision: Decision, prices: UniverseMapping
    ) -> Plan[TPlanOp_co]: ...


@dataclass(frozen=True, slots=True)
class TargetWeightsOp(PlanOp):
    weights: Mapping[str, float]
    cash: float | None
    fill_cash: bool


@dataclass(frozen=True, slots=True)
class TargetHoldingsOp(PlanOp):
    holdings: Mapping[str, int]
    cash: float | None
    fill_cash: bool


@dataclass(frozen=True, slots=True)
class MakeTradeOp(PlanOp):
    trade: TradeOrder


@dataclass(frozen=True, slots=True)
class ReallocateOp(PlanOp):
    inner: ReallocateDecision


@dataclass(frozen=True, slots=True)
class Plan[TPlanOp_co]:
    steps: tuple[TPlanOp_co, ...]


TargettingOp = TargetWeightsOp | TargetHoldingsOp
