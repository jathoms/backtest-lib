from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Mapping
from typing import (
    TypeVar,
    abstractmethod,
)

from backtest_lib.market import get_mapping_type_from_backend
from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.universe.vector_mapping import VectorMapping

Scalar = TypeVar("Scalar", float, int)
Other_scalar = TypeVar("Other_scalar", float, int)


class UniverseMapping[Scalar: (float, int)](VectorMapping[str, Scalar], ABC):
    @property
    def plot(self) -> UniverseMappingPlotAccessor: ...

    @abstractmethod
    def __truediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def __rtruediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> UniverseMapping[float]: ...

    @abstractmethod
    def floor(self) -> UniverseMapping[int]: ...

    @abstractmethod
    def truncate(self) -> UniverseMapping[int]: ...


def make_universe_mapping[T: (int, float)](
    m: Mapping[str, T],
    universe: Iterable[str],
    constructor_backend: str = "polars",
) -> UniverseMapping[T]:
    if not isinstance(m, UniverseMapping) and isinstance(m, Mapping):
        backend_mapping_type = get_mapping_type_from_backend(constructor_backend)
        # TODO: check if this dictionary collection is slow.
        # also, this doesn't check that securities passed in via the mapping are
        # actually part of the universe. checking for that would be quite slow,
        # so we avoid it for now.
        return backend_mapping_type.from_vectors(
            universe,
            (m.get(k, 0.0) for k in universe),
        )
    else:
        return m
