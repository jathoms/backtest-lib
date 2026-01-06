from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Protocol, Self, TypeVar

from backtest_lib.engine.decision import Decision, DecisionBase, TradeDirection
from backtest_lib.market import get_mapping_type_from_mapping
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


@dataclass(frozen=True)
class Trades:
    trades: tuple[TradeOrder, ...]
    security_alignment: tuple[str, ...]
    backend_mapping_type: type[UniverseMapping]

    @staticmethod
    def from_inputs(
        trades: Iterable[TradeOrder],
        *,
        security_alignment: Iterable[str],
        backend: str,
    ) -> Self:
        return Trades(
            trades=tuple(trades),
            security_alignment=tuple(security_alignment),
            backend_mapping_type=get_mapping_type_from_mapping(backend),
        )

    @cached_property
    def position_delta(self) -> UniverseMapping[int]:
        zeros = self.backend_mapping_type.from_vectors(
            self.security_alignment, (0 for _ in range(len(self.security_alignment)))
        ).floor()
        batched_trades: dict[str, int] = defaultdict(int)
        for t in self.trades:
            batched_trades[t.security] += t.signed_qty

        return zeros + batched_trades

    def total_cost(self) -> float:
        return sum(trade.cost() for trade in self.trades)

    def with_universe(self, universe: tuple[str, ...]) -> Self:
        return replace(self, security_alignment=universe)


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
class MakeTradesOp(PlanOp):
    trades: Trades


@dataclass(frozen=True, slots=True)
class MakeTradeOp(PlanOp):
    trade: TradeOrder


@dataclass(frozen=True, slots=True)
class Plan[TPlanOp_co]:
    steps: tuple[TPlanOp_co, ...]

    def __len__(self):
        return len(self.steps)


HOLD_PLAN = Plan(())

TargettingOp = TargetWeightsOp | TargetHoldingsOp
