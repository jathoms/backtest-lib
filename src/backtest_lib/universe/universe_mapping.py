from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
)

from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.universe.vector_mapping import VectorMapping

Scalar = TypeVar("Scalar", float, int)


@runtime_checkable
class UniverseMapping(VectorMapping[str, Scalar], Protocol[Scalar]):
    @property
    def plot(self) -> UniverseMappingPlotAccessor: ...
