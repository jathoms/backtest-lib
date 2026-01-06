from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"


class DecisionBase:
    def __add__(self: Decision, other: Decision) -> DecisionBase:
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
    fill_cash: bool
    cash: float | None = None


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision(DecisionBase):
    target_weights: Mapping[str, float]
    fill_cash: bool
    cash: float | None = None


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
    holdings: Mapping[str, int], fill_cash: bool = False
) -> TargetHoldingsDecision:
    return TargetHoldingsDecision(target_holdings=holdings, fill_cash=fill_cash)


def target_weights(
    weights: Mapping[str, float], fill_cash: bool = False
) -> TargetWeightsDecision:
    return TargetWeightsDecision(target_weights=weights, fill_cash=fill_cash)


def combine(*decisions: DecisionBase) -> DecisionBase:
    return sum(decisions, start=hold())


def trade(
    direction: str | TradeDirection, qty: int, security: str
) -> MakeTradeDecision:
    direction = TradeDirection(direction)
    return MakeTradeDecision(direction=direction, qty=qty, security=security)
