from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Protocol, Self

from backtest_lib.decision.plan import Plan, Trades

from backtest_lib.market import MarketView, get_mapping_type_from_mapping
from backtest_lib.portfolio import Portfolio
from backtest_lib.universe.universe_mapping import UniverseMapping


class PlanExecutor(Protocol):
    def execute(self, plan: Plan) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    fees: float
    slippage: float

    def __post_init__(self) -> None:
        if self.fees < 0 or self.slippage < 0:
            raise ValueError("Costs must be non-negative.")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    before: Portfolio
    after: Portfolio
    costs: CostBreakdown
    fills: Trades  # realized trades (may differ)
    warnings: tuple[str, ...] = ()
