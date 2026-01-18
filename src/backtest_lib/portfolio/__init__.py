from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from backtest_lib.market import get_mapping_type_from_backend
from backtest_lib.market.polars_impl import PolarsUniverseMapping
from backtest_lib.universe.universe_mapping import (
    UniverseMapping,
    make_universe_mapping,
)

Quantity = int
FractionalQuantity = float
Weight = float


logger = logging.getLogger(__name__)


class PortfolioBase[H: (float, int)]:
    """PLACEHOLDER"""

    holdings: UniverseMapping[H]
    cash: float = 0
    total_value: float
    universe: tuple[str, ...]
    _backend: str = "polars"

    def __init__(
        self,
        universe: Iterable[str],
        holdings: UniverseMapping[H] | Mapping[str, H],
        cash: float,
        total_value: float,
        constructor_backend: str = "polars",
    ):
        universe_tup = tuple(universe)
        self.holdings = make_universe_mapping(
            m=holdings,
            universe=universe_tup,
            constructor_backend=constructor_backend,
        )
        self.cash = cash
        self.total_value = total_value
        self.universe = universe_tup
        self._backend = constructor_backend

    @abstractmethod
    def into_weighted(
        self, prices: UniverseMapping | None = None
    ) -> WeightedPortfolio: ...

    @abstractmethod
    def into_quantities(
        self, prices: UniverseMapping | None = None
    ) -> QuantityPortfolio: ...

    @abstractmethod
    def into_quantities_fractional(
        self, prices: UniverseMapping | None = None
    ) -> FractionalQuantityPortfolio: ...


class QuantityPortfolio(PortfolioBase[Quantity]):
    """PLACEHOLDER"""

    def into_weighted(
        self, prices: UniverseMapping[float] | None = None
    ) -> WeightedPortfolio:
        if prices is None:
            raise ValueError(
                "Prices must be passed to convert from a "
                "quantity portfolio to a weighted portfolio"
            )
        values = self.holdings * prices
        total_value = values.sum() + self.cash
        weights = values / total_value
        cash_weight = self.cash / total_value
        return WeightedPortfolio(
            cash=cash_weight,
            holdings=weights,
            total_value=total_value,
            universe=self.universe,
            constructor_backend=self._backend,
        )

    def into_quantities(self, prices=None) -> QuantityPortfolio:
        del prices
        return self

    def into_quantities_fractional(self, prices=None) -> FractionalQuantityPortfolio:
        del prices
        return FractionalQuantityPortfolio(
            holdings=self.holdings * 1.0,
            cash=self.cash,
            total_value=self.total_value,
            universe=self.universe,
            constructor_backend=self._backend,
        )


class FractionalQuantityPortfolio(PortfolioBase[FractionalQuantity]):
    """PLACEHOLDER"""

    def into_weighted(self, prices: UniverseMapping | None = None) -> WeightedPortfolio:
        if prices is None:
            raise ValueError(
                "Prices must be passed to convert from a "
                "fractional quantity portfolio to a weighted portfolio"
            )
        values = self.holdings * prices
        total_value = values.sum() + self.cash
        weights = values / total_value
        cash_weight = self.cash / total_value
        return WeightedPortfolio(
            cash=cash_weight,
            holdings=weights,
            universe=self.universe,
            total_value=total_value,
            constructor_backend=self._backend,
        )

    def into_quantities(self, prices=None) -> QuantityPortfolio:
        del prices
        return QuantityPortfolio(
            holdings=self.holdings.floor(),
            cash=self.cash,
            total_value=self.total_value,
            universe=self.universe,
            constructor_backend=self._backend,
        )

    def into_quantities_fractional(self, prices=None) -> FractionalQuantityPortfolio:
        del prices
        return self


class WeightedPortfolio(PortfolioBase[Weight]):
    """PLACEHOLDER"""

    def into_weighted(self, prices=None) -> WeightedPortfolio:
        del prices
        return self

    def into_quantities(
        self, prices: UniverseMapping | None = None
    ) -> QuantityPortfolio:
        if prices is None:
            raise ValueError(
                "Prices must be passed to convert from a "
                "weighted portfolio to a quantity portfolio"
            )
        target_qtys = (self.total_value * self.holdings) / prices
        qtys = target_qtys.floor()
        spent = (qtys * prices).sum()
        cash_value = self.total_value - spent
        return QuantityPortfolio(
            cash=cash_value,
            holdings=qtys,
            total_value=self.total_value,
            universe=self.universe,
            constructor_backend=self._backend,
        )

    def into_quantities_fractional(
        self, prices: UniverseMapping | None = None
    ) -> FractionalQuantityPortfolio:
        if prices is None:
            raise ValueError(
                "Prices must be passed to convert from a "
                "weighted portfolio to a fractional quantity portfolio"
            )
        cash = self.total_value * self.cash
        holdings = (self.total_value * self.holdings) / prices
        return FractionalQuantityPortfolio(
            cash=cash,
            holdings=holdings,
            total_value=self.total_value,
            universe=self.universe,
            constructor_backend=self._backend,
        )

    def into_long_only(self) -> WeightedPortfolio:
        if isinstance(self.holdings, PolarsUniverseMapping):
            import polars as pl

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
            return WeightedPortfolio(
                cash=self.cash,
                holdings=new_holdings,
                universe=self.universe,
                total_value=self.total_value,
                constructor_backend=self._backend,
            )
        else:
            raise NotImplementedError()

    def indexed_over(self, full_universe: Iterable[str]) -> WeightedPortfolio:
        return uniform_portfolio(full_universe, self.holdings.keys())


def uniform_portfolio(
    full_universe: Iterable[str],
    tradable_universe: Iterable[str] | None = None,
    value: float = 1.0,
    backend: str = "polars",
) -> WeightedPortfolio:
    """PLACEHOLDER"""
    if tradable_universe is None:
        tradable_universe = full_universe
    full_tup = tuple(full_universe)
    if not isinstance(tradable_universe, set):
        tradable_universe = set(tradable_universe)
    uniform_allocation_weight = 1.0 / min(len(tradable_universe), len(full_tup))
    return WeightedPortfolio(
        cash=0,
        universe=full_tup,
        holdings=get_mapping_type_from_backend(backend).from_vectors(
            full_tup,
            (
                uniform_allocation_weight if sec in tradable_universe else 0.0
                for sec in full_tup
            ),
        ),
        total_value=value,
        constructor_backend=backend,
    )


@dataclass(frozen=True, slots=True)
class Cash:
    value: float

    def materialize(self, universe: Iterable[str], backend: str) -> WeightedPortfolio:
        mapping_type = get_mapping_type_from_backend(backend)
        universe_tup = tuple(universe)
        return WeightedPortfolio(
            holdings=mapping_type.from_vectors(
                universe_tup, (0.0 for _ in universe_tup)
            ),
            cash=1.0,
            constructor_backend=backend,
            universe=universe_tup,
            total_value=self.value,
        )


def cash(value: float):
    return Cash(value=value)


Portfolio = WeightedPortfolio | QuantityPortfolio | FractionalQuantityPortfolio
