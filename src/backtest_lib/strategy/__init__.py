from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Generic, Protocol, TypeVar


from backtest_lib.market import MarketView
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Price, Universe, UniverseMapping

Quantity = int
FractionalQuantity = float
Weight = float

H = TypeVar("H", bound=Real, covariant=True)


@dataclass(frozen=True)
class Portfolio(Generic[H]):
    cash: H
    holdings: UniverseMapping[H]  # asset -> quantity

    def into_weighted(self, prices: UniverseMapping[Price]) -> WeightedPortfolio:
        securities = list(self.holdings.keys())
        values = (self.holdings[security] * prices[security] for security in securities)
        total_value = sum(values) + self.cash
        weights = (value / total_value for value in values)
        cash_weight = self.cash / total_value
        asset_weights = dict(zip(securities, weights))
        return WeightedPortfolio(cash=cash_weight, weights=asset_weights)


@dataclass(frozen=True)
class WeightedPortfolio:
    cash: Weight  # cash weight in [0, 1]
    weights: UniverseMapping[Weight]

    def into_quantities(self, prices: UniverseMapping, total_value: Price) -> Portfolio:
        cash_weight = total_value * self.cash
        holdings = {
            sec: ((total_value * self.weights[sec]) / prices[sec])
            for sec in self.weights.keys()
        }

        return Portfolio(cash=cash_weight, holdings=holdings)


@dataclass(frozen=True)
class Decision:
    target: WeightedPortfolio
    notes = UniverseMapping[Any] | None


class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_portfolio: Portfolio,
        market: MarketView,
        # schedule: Schedule, # for seeing where we are in the rebalance schedule
        ctx: StrategyContext | None,
    ) -> Decision: ...
