"""Decision types and constructors for strategy output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ReallocationMode(StrEnum):
    """Modes for allocating out of and into security sets."""

    EQUAL_OUT_EQUAL_IN = "equal_out_equal_in"
    PRO_RATA_OUT_EQUAL_IN = "pro_rata_out_equal_in"


class TradeDirection(StrEnum):
    """Direction for trade decisions created by :func:`trade`."""

    BUY = "buy"
    SELL = "sell"


class DecisionBase:
    """Base type for all decision objects.

    Decisions are the output language of a
    :class:`~backtest_lib.strategy.Strategy` and can be combined with ``+`` to
    form composite decisions.
    """

    def __add__(self, other: Decision) -> Decision:
        """Combine two decisions into a composite decision."""
        if isinstance(self, HoldDecision):
            return other
        if isinstance(other, HoldDecision):
            return self
        left = self.decisions if isinstance(self, CompositeDecision) else (self,)
        right = other.decisions if isinstance(other, CompositeDecision) else (other,)
        return CompositeDecision(left + right)


@dataclass(frozen=True, slots=True)
class MakeTradeDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.trade`."""

    direction: TradeDirection
    qty: int
    security: str

    def __post_init__(self):
        """Validate that the trade quantity is non-negative."""
        if self.qty < 0:
            raise ValueError("qty must be non-negative.")


@dataclass(frozen=True, slots=True)
class CompositeDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.combine`."""

    decisions: tuple[Decision, ...]


@dataclass(frozen=True, slots=True)
class TargetHoldingsDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.target_holdings`."""

    target_holdings: Mapping[str, int]
    fill_cash: bool
    cash: float | None = None


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.target_weights`."""

    target_weights: Mapping[str, float]
    fill_cash: bool
    cash: float | None = None


@dataclass(frozen=True, slots=True)
class HoldDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.hold`."""

    pass


@dataclass(frozen=True, slots=True)
class ReallocateDecision(DecisionBase):
    """Decision returned by :func:`~backtest_lib.engine.decision.reallocate`."""

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
    """Create a reallocation decision across security sets.

    This is part of the strategy decision language. The engine interprets this
    decision as a transfer of exposure from ``out_of`` securities into ``into``
    securities according to the chosen ReallocationMode.

    Args:
        fraction: Fraction of holdings to reallocate.
        out_of: Securities to reduce positions in.
        into: Securities to increase positions in.
        mode: Allocation mode for distributing sales and buys.

    Returns:
        ReallocateDecision describing the move.

    """
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


#: The decision type returned from a strategy, built from decision functions.
Decision = (
    HoldDecision
    | MakeTradeDecision
    | TargetWeightsDecision
    | TargetHoldingsDecision
    | ReallocateDecision
    | CompositeDecision
)


def hold() -> HoldDecision:
    """Create a decision that makes no trades.

    This is the neutral element of the decision language. When combined with
    another decision, the other decision is returned unchanged, mirroring
    :func:`~backtest_lib.engine.decision.combine`.
    """
    return HoldDecision()


def target_holdings(
    holdings: Mapping[str, int], fill_cash: bool = False
) -> TargetHoldingsDecision:
    """Create a decision targeting discrete holdings.

    This is part of the strategy output language. Use it when your strategy
    expresses desired positions as share counts.

    Args:
        holdings: Desired integer holdings per security.
        fill_cash: Whether to allocate remaining cash automatically.

    Returns:
        TargetHoldingsDecision for the target holdings.

    """
    return TargetHoldingsDecision(target_holdings=holdings, fill_cash=fill_cash)


def target_weights(
    weights: Mapping[str, float], fill_cash: bool = False
) -> TargetWeightsDecision:
    """Create a decision targeting portfolio weights.

    This is part of the strategy output language. Use it when your strategy
    expresses desired positions as weights that should sum with cash.

    Args:
        weights: Desired weights per security.
        fill_cash: Whether to allocate remaining cash automatically.

    Returns:
        TargetWeightsDecision for the target weights.

    """
    return TargetWeightsDecision(target_weights=weights, fill_cash=fill_cash)


def combine(*decisions: Decision) -> Decision:
    """Combine multiple decisions into a single composite decision.

    This is the composition operator for the decision language. It mirrors the
    ``+`` operator implemented by the decision base class.

    Example:
        >>> from backtest_lib.engine.decision import combine, hold, target_weights
        >>> d1 = hold()
        >>> d2 = target_weights({"AAPL": 1.0})
        >>> d1 + d2 == combine(d1, d2)
        True
    """
    return sum(decisions, start=hold())


def trade(
    direction: str | TradeDirection, qty: int, security: str
) -> MakeTradeDecision:
    """Create a decision to trade a specific quantity.

    This is part of the strategy output language for explicit trades. Use it
    when you want a single-buy/sell instruction rather than a target portfolio
    expressed via :func:`~backtest_lib.engine.decision.target_weights`.

    Args:
        direction: Buy or sell direction.
        qty: Quantity to trade.
        security: Security identifier to trade.

    Returns:
        MakeTradeDecision describing the trade.

    """
    direction = TradeDirection(direction)
    return MakeTradeDecision(direction=direction, qty=qty, security=security)
