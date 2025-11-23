from __future__ import annotations

from typing import Protocol, TypeVar, abstractmethod, runtime_checkable

from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.universe.vector_mapping import VectorMapping
from backtest_lib.universe.vector_ops import VectorOps

Scalar = TypeVar("Scalar", float, int)
Other_scalar = TypeVar("Other_scalar", float, int)


@runtime_checkable
class UniverseMapping(VectorMapping[str, Scalar], Protocol[Scalar]):
    @property
    def plot(self) -> UniverseMappingPlotAccessor: ...

    @abstractmethod
    def __truediv__(
        self, other: VectorOps | Other_scalar
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorOps | Other_scalar
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def floor(self) -> UniverseMapping[int]: ...
