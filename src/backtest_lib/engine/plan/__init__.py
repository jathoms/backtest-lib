from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from backtest_lib.engine.decision import Decision, TradeDirection
from backtest_lib.portfolio import QuantityPortfolio, WeightedPortfolio
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


TPlanOp_co = TypeVar("TPlanOp_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)


class PlanGenerator[T_co](Protocol):
    def generate_plan(
        self, decision: Decision, prices: UniverseMapping
    ) -> Plan[T_co]: ...


@dataclass(frozen=True, slots=True)
class TargetWeightsOp(PlanOp):
    weights: Mapping[str, float]
    cash: float

    def to_portfolio(self, total_value: float, backend: str) -> WeightedPortfolio:
        return WeightedPortfolio(
            holdings=self.weights,
            cash=self.cash,
            total_value=total_value,
            constructor_backend=backend,
        )


@dataclass(frozen=True, slots=True)
class TargetHoldingsOp(PlanOp):
    holdings: Mapping[str, int]
    cash: float

    def to_portfolio(self, total_value: float, backend: str) -> QuantityPortfolio:
        return QuantityPortfolio(
            holdings=self.holdings,
            cash=self.cash,
            total_value=total_value,
            constructor_backend=backend,
        )


@dataclass(frozen=True, slots=True)
class MakeTradeOp(PlanOp):
    trade: TradeOrder


@dataclass(frozen=True, slots=True)
class Plan[TPlanOp_co]:
    steps: Iterator[TPlanOp_co]


TargettingOp = TargetWeightsOp | TargetHoldingsOp
