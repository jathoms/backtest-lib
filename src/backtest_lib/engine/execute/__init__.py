from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Protocol, Self, TypeVar

from backtest_lib.engine.plan import Plan, PlanOp, Trades
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


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    before: Portfolio
    after: Portfolio
    costs: CostBreakdown
    fills: Trades | None
    warnings: tuple[str, ...] = ()
