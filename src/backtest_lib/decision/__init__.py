from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from itertools import chain
from typing import Any, Literal, Self, override

from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"


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


class Decision:
    def __add__(self, other: Decision) -> CompositeDecision:
        left = self.decisions if isinstance(self, CompositeDecision) else (self,)
        right = other.decisions if isinstance(other, CompositeDecision) else (other,)
        return CompositeDecision(left + right)

    def __radd__(self, other: Any) -> CompositeDecision | Decision:
        # allow sum([...]) with start=0 (default). otherwise NotImplemented.
        if other == 0:
            return self
        return NotImplemented


@dataclass(frozen=True, slots=True)
class MakeTradeDecision(Decision):
    direction: TradeDirection
    qty: int
    security: str

    def __post_init__(self):
        if self.qty < 0:
            raise ValueError("qty must be non-negative.")


@dataclass(frozen=True, slots=True)
class CompositeDecision(Decision):
    decisions: tuple[Decision, ...]


@dataclass(frozen=True, slots=True)
class TargetHoldingsDecision(Decision):
    target_holdings: Mapping[str, float]
    cash: float = 0


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision(Decision):
    target_weights: Mapping[str, float]
    cash: float = 0


@dataclass(frozen=True, slots=True)
class HoldDecision(Decision): ...


def hold() -> HoldDecision:
    return HoldDecision()


def target_holdings(
    holdings: Mapping[str, float], cash: float = 0
) -> TargetHoldingsDecision:
    return TargetHoldingsDecision(target_holdings=holdings, cash=cash)


def target_weights(
    weights: Mapping[str, float], cash: float = 0
) -> TargetWeightsDecision:
    return TargetWeightsDecision(target_weights=weights, cash=cash)


def combine(*decisions: Decision) -> Decision:
    return sum(decisions, start=hold())


def trade(
    direction: str | TradeDirection, qty: int, security: str
) -> MakeTradeDecision:
    direction = TradeDirection(direction)
    return MakeTradeDecision(direction=direction, qty=qty, security=security)
