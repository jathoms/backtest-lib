from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
from collections.abc import Sequence
from backtest_lib.universe.vector_mapping import VectorMappingConstructor


from backtest_lib.market import MarketView
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Price, Universe, UniverseMapping
from backtest_lib.universe.vector_mapping import VectorMapping

Quantity = int
FractionalQuantity = float
Weight = float

H = TypeVar("H", bound=float)
MappingType = TypeVar("MappingType", bound=VectorMapping)


@dataclass(frozen=True)
class Portfolio(Generic[H, MappingType]):
    cash: H
    holdings: UniverseMapping[H]  # asset -> quantity
    _mapping_cls: VectorMappingConstructor[MappingType]

    @classmethod
    def from_raw(
        cls: type[Portfolio[H, MappingType]],
        *,
        cash: H,
        keys: Sequence[str],
        values: Sequence[H],
        mapping_cls: VectorMappingConstructor[MappingType],
    ) -> Portfolio[H, MappingType]:
        return cls(
            cash=cash,
            holdings=mapping_cls.from_vectors(keys, values),
            _mapping_cls=mapping_cls,
        )

    def into_weighted(self, prices: UniverseMapping[Price]) -> WeightedPortfolio:
        securities = tuple(self.holdings.keys())
        values = self.holdings * prices
        total_value = values.sum() + self.cash
        weights = values / total_value
        cash_weight = self.cash / total_value
        return WeightedPortfolio.from_raw(
            cash=cash_weight,
            keys=securities,
            values=list(weights.values()),
            mapping_cls=self._mapping_cls,
        )


@dataclass(frozen=True)
class WeightedPortfolio(Generic[MappingType]):
    cash: Weight  # cash weight in [0, 1]
    weights: UniverseMapping[Weight]
    _mapping_cls: VectorMappingConstructor[MappingType]

    @classmethod
    def from_raw(
        cls: type[WeightedPortfolio[MappingType]],
        *,
        cash: float,
        keys: Sequence[str],
        values: Sequence[float],
        mapping_cls: VectorMappingConstructor[MappingType],
    ) -> WeightedPortfolio[MappingType]:
        return cls(
            cash=cash,
            weights=mapping_cls.from_vectors(keys, values),
            _mapping_cls=mapping_cls,
        )

    def into_quantities(
        self, prices: UniverseMapping[Price], total_value: Price
    ) -> Portfolio:
        cash_weight = total_value * self.cash
        holdings = (total_value * self.weights) / prices

        return Portfolio.from_raw(
            cash=cash_weight,
            keys=list(self.weights.keys()),
            values=list(holdings.values()),
            mapping_cls=self._mapping_cls,
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
