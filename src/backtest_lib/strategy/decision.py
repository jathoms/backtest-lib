from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping

type TradeDirection = Literal["buy"] | Literal["sell"]


@dataclass(frozen=True)
class TradeInstruction:
    direction: TradeDirection
    qty: int
    security: str
    price: float

    @property
    def signed_qty(self) -> int:
        return -self.qty if self.direction == "sell" else self.qty

    def cost(self) -> float:
        return self.signed_qty * self.price


def _calculate_total_position_delta(
    trades: Iterable[TradeInstruction],
    security_alignment: Sequence[str],
    mapping_type: type[UniverseMapping],
) -> UniverseMapping[int]:
    zeros = mapping_type.from_vectors(
        security_alignment, [0] * len(security_alignment)
    ).floor()
    batched_trades: dict[str, int] = defaultdict(int)
    for t in trades:
        batched_trades[t.security] += t.signed_qty

    return zeros + batched_trades


class Trades:
    _trades: Sequence[TradeInstruction]
    _security_alignment: Sequence[str]
    _backend_mapping_type: type[UniverseMapping]

    def __init__(
        self, trades: Sequence[TradeInstruction], *, security_alignment, backend: str
    ):
        self._trades = trades
        self._security_alignment = security_alignment
        self._backend_mapping_type = get_mapping_type_from_mapping(backend)

    @cached_property
    def position_delta(self) -> UniverseMapping[int]:
        return _calculate_total_position_delta(
            self._trades, self._security_alignment, self._backend_mapping_type
        )

    def total_cost(self) -> float:
        return sum(trade.cost() for trade in self._trades)


@dataclass(frozen=True, slots=True)
class MakeTradesDecision:
    trades: Trades


@dataclass(frozen=True, slots=True)
class AlterPositionsDecision:
    adjustments: UniverseMapping[int]


@dataclass(frozen=True, slots=True)
class TargetWeightsDecision:
    targets: UniverseMapping[float]


Decision = MakeTradesDecision | AlterPositionsDecision | TargetWeightsDecision
