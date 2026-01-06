from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"


class DecisionBase:
    def __add__(self, other: Decision) -> DecisionBase:
        assert not isinstance(self, DecisionBase)
        if isinstance(self, HoldDecision):
            return other
        if isinstance(other, HoldDecision):
            return self
        left = self.decisions if isinstance(self, CompositeDecision) else (self,)
        right = other.decisions if isinstance(other, CompositeDecision) else (other,)
        return CompositeDecision(left + right)


@dataclass(frozen=True, slots=True)
class MakeTradeDecision(DecisionBase):
    direction: TradeDirection
    qty: int
    security: str

    def __post_init__(self):
        if self.qty < 0:
            raise ValueError("qty must be non-negative.")


@dataclass(frozen=True, slots=True)
class CompositeDecision(DecisionBase):
    decisions: tuple[Decision, ...]


@dataclass(frozen=True, slots=True)
class TargetHoldingsDecision(DecisionBase):
    target_holdings: Mapping[str, int]
    cash: float = 0


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision(DecisionBase):
    target_weights: Mapping[str, float]
    cash: float = 0


@dataclass(frozen=True, slots=True)
class HoldDecision(DecisionBase):
    pass


Decision = (
    HoldDecision
    | MakeTradeDecision
    | TargetWeightsDecision
    | TargetHoldingsDecision
    | CompositeDecision
)


def hold() -> HoldDecision:
    return HoldDecision()


def target_holdings(
    holdings: Mapping[str, int], cash: float = 0
) -> TargetHoldingsDecision:
    return TargetHoldingsDecision(target_holdings=holdings, cash=cash)


def target_weights(
    weights: Mapping[str, float], cash: float = 0
) -> TargetWeightsDecision:
    return TargetWeightsDecision(target_weights=weights, cash=cash)


def combine(*decisions: DecisionBase) -> DecisionBase:
    return sum(decisions, start=hold())


def trade(
    direction: str | TradeDirection, qty: int, security: str
) -> MakeTradeDecision:
    direction = TradeDirection(direction)
    return MakeTradeDecision(direction=direction, qty=qty, security=security)
