from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import (
    TypeVar,
    abstractmethod,
)

from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.universe.vector_mapping import VectorMapping
from backtest_lib.universe.vector_ops import VectorOps

Scalar = TypeVar("Scalar", float, int)
Other_scalar = TypeVar("Other_scalar", float, int)


class UniverseMapping[Scalar: (float, int)](VectorMapping[str, Scalar], ABC):
    @property
    def plot(self) -> UniverseMappingPlotAccessor: ...

    @abstractmethod
    def __truediv__(
        self, other: VectorOps[Other_scalar] | Other_scalar | Mapping
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorOps[Other_scalar] | Other_scalar | Mapping
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def floor(self) -> UniverseMapping[int]: ...
