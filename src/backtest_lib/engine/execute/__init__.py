from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Protocol, Self, TypeVar

from backtest_lib.engine.plan import Plan, TradeOrder
from backtest_lib.market import MarketView, get_mapping_type_from_mapping
from backtest_lib.portfolio import Portfolio
from backtest_lib.universe.universe_mapping import UniverseMapping

TPlanOp_contra = TypeVar("TPlanOp_contra", contravariant=True)


class PlanExecutor[TPlanOp_contra](Protocol):
    def execute_plan(
        self,
        plan: Plan[TPlanOp_contra],
        portfolio: Portfolio,
        prices: UniverseMapping,
        market: MarketView,
    ) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    fees: float
    slippage: float

    def __post_init__(self) -> None:
        if self.fees < 0 or self.slippage < 0:
            raise ValueError("Costs must be non-negative.")


NO_COST = CostBreakdown(fees=0, slippage=0)


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


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    before: Portfolio
    after: Portfolio
    costs: CostBreakdown
    fills: Trades | None
    warnings: tuple[str, ...] = ()
