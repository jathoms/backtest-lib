from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from typing import Literal, Self

from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class TradeInstruction:
    direction: TradeDirection
    qty: int
    security: str


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
    ) -> Trades:
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


class Decision(ABC):
    @abstractmethod
    def implied_universe(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def with_universe(self, universe: tuple[str, ...]) -> Self: ...


@dataclass(frozen=True, slots=True)
class MakeTradesDecision(Decision):
    trades: Trades

    def implied_universe(self) -> tuple[str, ...]:
        return self.trades.security_alignment

    def with_universe(self, universe: tuple[str, ...]) -> Self:
        return replace(self, trades=self.trades.with_universe(universe))


@dataclass(frozen=True, slots=True)
class AlterPositionsDecision(Decision):
    adjustments: UniverseMapping[int]

    def implied_universe(self) -> tuple[str, ...]:
        return tuple(self.adjustments.keys())

    def with_universe(self, universe: tuple[str, ...]) -> Self:
        return replace(
            self, adjustments=self.adjustments + {sec: 0 for sec in universe}
        )


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision(Decision):
    target_weights: UniverseMapping[float]
    cash: float = 0

    def implied_universe(self) -> tuple[str, ...]:
        return tuple(self.target_weights.keys())

    def with_universe(self, universe: tuple[str, ...]) -> Self:
        return replace(
            self, target_weights=self.target_weights + {sec: 0 for sec in universe}
        )


def make_trades() -> MakeTradesDecision: ...


def trade(direction: str | TradeDirection, qty: int, security: str) -> TradeInstruction:
    direction = TradeDirection(direction)
    return TradeInstruction(direction=direction, qty=qty, security=security)
