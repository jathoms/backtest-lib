from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


from backtest_lib.market import MarketView
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Price, Universe, UniverseMapping
from backtest_lib.universe.vector_mapping import VectorMapping

Quantity = int
FractionalQuantity = float
Weight = float

H = TypeVar("H", int, float)
MappingType = TypeVar("MappingType", bound=VectorMapping)


@dataclass(frozen=True)
class Portfolio(Generic[H, MappingType]):
    cash: float
    holdings: UniverseMapping[H]  # asset -> quantity


class QuantityPortfolio(Portfolio[Quantity, MappingType], Generic[MappingType]):
    def into_weighted(self, prices: UniverseMapping[Price]) -> WeightedPortfolio:
        values = self.holdings * prices
        total_value = values.sum() + self.cash
        weights = values / total_value
        cash_weight = self.cash / total_value
        return WeightedPortfolio(
            cash=cash_weight,
            holdings=weights,
        )


class FractionalQuantityPortfolio(
    Portfolio[FractionalQuantity, MappingType], Generic[MappingType]
):
    def into_weighted(self, prices: UniverseMapping[Price]) -> WeightedPortfolio:
        values = self.holdings * prices
        total_value = values.sum() + self.cash
        weights = values / total_value
        cash_weight = self.cash / total_value
        return WeightedPortfolio(
            cash=cash_weight,
            holdings=weights,
        )


@dataclass(frozen=True)
class WeightedPortfolio(Portfolio[Weight, MappingType], Generic[MappingType]):
    def into_quantities(
        self, prices: UniverseMapping[Price], total_value: Price
    ) -> QuantityPortfolio:
        target_qtys = (total_value * self.holdings) / prices
        qtys = target_qtys.floor()
        spent = (qtys * prices).sum()
        cash_value = total_value - spent
        return QuantityPortfolio(
            cash=cash_value,
            holdings=qtys,
        )

    def into_quantities_fractional(
        self, prices: UniverseMapping[Price], total_value: Price
    ) -> FractionalQuantityPortfolio:
        cash = total_value * self.cash
        holdings = (total_value * self.holdings) / prices
        return FractionalQuantityPortfolio(
            cash=cash,
            holdings=holdings,
        )


@dataclass(frozen=True)
class Decision:
    target: WeightedPortfolio
    notes = UniverseMapping[Any] | None


class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_portfolio: WeightedPortfolio,
        market: MarketView,
        # schedule: Schedule, # for seeing where we are in the rebalance schedule
        ctx: StrategyContext | None,
    ) -> Decision: ...
