from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Self, TypeVar, cast

import polars as pl

from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.market.polars_impl import SeriesUniverseMapping
from backtest_lib.strategy.decision import (
    AlterPositionsDecision,
    Decision,
    MakeTradesDecision,
    TargetWeightsDecision,
)
from backtest_lib.universe.universe_mapping import UniverseMapping

Quantity = int
FractionalQuantity = float
Weight = float

H = TypeVar("H", float, int)


class NegativeCashException(Exception): ...


class Portfolio[H: (float, int)]:
    """PLACEHOLDER"""

    holdings: UniverseMapping[H]
    cash: float = 0
    total_value: float
    _backend: str = "polars"

    def __init__(
        self,
        holdings: UniverseMapping[H] | Mapping[str, H],
        cash: float = 0,
        total_value: float = 0,
        constructor_backend="polars",
    ):
        self.holdings = make_universe_mapping(holdings, constructor_backend)
        self.cash = cash
        self.total_value = total_value
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

    def after_decision(self, decision: Decision, prices: UniverseMapping) -> Self:
        match decision:
            case MakeTradesDecision(trades=trades):
                pos_delta = trades.position_delta
                decision_cost = trades.total_cost()
                new_cash = self.cash - decision_cost

                if new_cash < 0:
                    # TODO: add settings for when this raises vs gives a warning etc.
                    raise NegativeCashException()

                qtys = self.into_quantities(prices=prices)
                new_holdings = qtys.holdings + pos_delta

                # TODO: This implicitly converts the user's portfolio into a
                # QuantityPortfolio when they return a MakeTradesDecision. Review
                return QuantityPortfolio(
                    holdings=new_holdings,
                    cash=new_cash,
                    total_value=self.total_value,
                    constructor_backend=self._backend,
                )

            case AlterPositionsDecision():
                ...
            case TargetWeightsDecision():
                ...
        return self


class QuantityPortfolio(Portfolio[Quantity]):
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
            constructor_backend=self._backend,
        )


class FractionalQuantityPortfolio(Portfolio[FractionalQuantity]):
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
        )

    def into_quantities(self, prices=None) -> QuantityPortfolio:
        del prices
        return QuantityPortfolio(
            holdings=self.holdings.floor(),
            cash=self.cash,
            total_value=self.total_value,
            constructor_backend=self._backend,
        )

    def into_quantities_fractional(self, prices=None) -> FractionalQuantityPortfolio:
        del prices
        return self


class WeightedPortfolio(Portfolio[Weight]):
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
        print(locals())
        return QuantityPortfolio(
            cash=cash_value,
            holdings=qtys,
            total_value=self.total_value,
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
            constructor_backend=self._backend,
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
    """PLACEHOLDER"""
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


def make_universe_mapping[T: (int, float)](
    m: Mapping[str, T], constructor_backend: str = "polars"
) -> UniverseMapping[T]:
    if not isinstance(m, UniverseMapping) and isinstance(m, Mapping):
        backend_mapping_type = get_mapping_type_from_mapping(constructor_backend)
        native_mapping = backend_mapping_type.from_vectors(
            # TODO: not sure what's going on with the typechecker here,
            # just manually casting .values() for now.
            tuple(m.keys()),
            tuple(cast(Iterable, m.values())),
        )
        return native_mapping
    else:
        return m
