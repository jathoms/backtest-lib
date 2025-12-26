from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Generic, TypeVar, cast

import polars as pl

from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.market.polars_impl import SeriesUniverseMapping
from backtest_lib.universe import Price
from backtest_lib.universe.universe_mapping import UniverseMapping
from backtest_lib.universe.vector_mapping import VectorMapping

Quantity = int
FractionalQuantity = float
Weight = float

H = TypeVar("H", float, int)
MappingType = TypeVar("MappingType", bound=VectorMapping)


class Portfolio(Generic[H, MappingType]):
    holdings: UniverseMapping[H]
    cash: float = 0

    def __init__(
        self,
        holdings: UniverseMapping[H] | Mapping[str, H],
        cash: float = 0,
        constructor_backend="polars",
    ):
        self.cash = cash

        if not isinstance(holdings, UniverseMapping) and isinstance(holdings, Mapping):
            backend_mapping_type = get_mapping_type_from_mapping(constructor_backend)
            native_mapping = backend_mapping_type.from_vectors(
                # TODO: not sure what's going on with the typechecker here,
                # just manually casting .values() for now.
                tuple(holdings.keys()),
                tuple(cast(Iterable, holdings.values())),
            )
            self.holdings = native_mapping
        else:
            self.holdings = holdings


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

    def into_long_only(self) -> WeightedPortfolio:
        if isinstance(self.holdings, SeriesUniverseMapping):
            # Assumes we want to keep our leverage ratio at 1
            df = pl.DataFrame({"w": self.holdings.as_series()})
            redistributed_weights = (
                df.select(
                    pl.col("w"),
                    pos_sum=pl.col("w").clip(lower_bound=0).sum().over(pl.lit(1)),
                    neg_mass=(-pl.col("w").clip(upper_bound=0).sum().over(pl.lit(1))),
                    exposure_ratio=(pl.col("w").abs().sum().over(pl.lit(1))),
                )
                .with_columns(
                    scale=pl.when(pl.col("pos_sum") > 0)
                    .then(1 + pl.col("neg_mass") / pl.col("pos_sum"))
                    .otherwise(0)
                )
                .with_columns(
                    redistributed=pl.when(pl.col("w") > 0)
                    .then(pl.col("w") * pl.col("scale"))
                    .otherwise(0)
                )
                .with_columns(
                    norm_redist_w=pl.col("redistributed") / pl.col("exposure_ratio")
                )
            )["norm_redist_w"]
            new_holdings = replace(self.holdings, _data=redistributed_weights)
            return WeightedPortfolio(cash=self.cash, holdings=new_holdings)
        else:
            raise NotImplementedError()

    def indexed_over(self, full_universe: Iterable[str]) -> WeightedPortfolio:
        return uniform_portfolio(full_universe, self.holdings.keys())


def uniform_portfolio(
    full_universe: Iterable[str], tradable_universe: Iterable[str] | None = None
) -> WeightedPortfolio:
    if tradable_universe is None:
        tradable_universe = full_universe
    if not isinstance(tradable_universe, set):
        tradable_universe = set(tradable_universe)
    uniform_allocation_weight = 1.0 / min(
        len(tradable_universe), len(list(full_universe))
    )
    return WeightedPortfolio(
        cash=0,
        holdings=SeriesUniverseMapping.from_names_and_data(
            tuple(full_universe),
            pl.Series(
                
                    uniform_allocation_weight if sec in tradable_universe else 0.0
                    for sec in full_universe
                
            ),
        ),
    )
