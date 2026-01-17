from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ReallocationMode(StrEnum):
    EQUAL_OUT_EQUAL_IN = "equal_out_equal_in"
    PRO_RATA_OUT_EQUAL_IN = "pro_rata_out_equal_in"


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"


class DecisionBase:
    def __add__(self: Decision, other: Decision) -> Decision:
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


@dataclass(frozen=True, slots=True)
class ReallocateDecision(DecisionBase):
    fraction: float
    from_securities: frozenset[str]
    to_securities: frozenset[str]
    mode: ReallocationMode


def reallocate(
    fraction: float,
    *,
    out_of: Iterable[str],
    into: Iterable[str],
    mode: Literal["pro_rata_out_equal_in", "equal_out_equal_in"]
    | ReallocationMode = ReallocationMode.EQUAL_OUT_EQUAL_IN,
) -> ReallocateDecision:
    mode = ReallocationMode(mode)
    if fraction < 0:
        raise ValueError("fraction must be non-negative")
    to_set = frozenset(into)
    if not to_set:
        raise ValueError("'to' must be non-empty")
    from_set = frozenset(out_of)
    if not from_set:
        raise ValueError("'out_of' must be non-empty")
    return ReallocateDecision(
        fraction=fraction,
        from_securities=from_set,
        to_securities=to_set,
        mode=mode,
    )


Decision = (
    HoldDecision
    | MakeTradeDecision
    | TargetWeightsDecision
    | TargetHoldingsDecision
    | ReallocateDecision
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


def combine(*decisions: Decision) -> Decision:
    return sum(decisions, start=hold())


def trade(
    direction: str | TradeDirection, qty: int, security: str
) -> MakeTradeDecision:
    direction = TradeDirection(direction)
    return MakeTradeDecision(direction=direction, qty=qty, security=security)
