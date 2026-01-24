"""Universe-aligned numeric mappings.

Universe mappings are vector mappings keyed by security identifiers. Backends
can use aligned key ordering to perform fast vectorized arithmetic. When two
mappings share the same universe ordering, operations are fastest; mismatched
ordering remains correct but incurs extra reindexing. See
``tests/benchmark/polars_impl/test_universe_mapping_benchmark.py`` for benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TypeVar

from backtest_lib.market._backends import _get_mapping_type_from_backend
from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.universe.vector_mapping import VectorMapping

Scalar = TypeVar("Scalar", float, int)
Other_scalar = TypeVar("Other_scalar", float, int)


class UniverseMapping[Scalar: (float, int)](VectorMapping[str, Scalar], ABC):
    """Vector mapping keyed by security identifiers.

    Implementations back this interface with dense vectors aligned to a universe
    ordering, enabling fast arithmetic when operands share the same ordering.
    """

    @property
    def plot(self) -> UniverseMappingPlotAccessor:
        """Return the plotting accessor for the mapping."""
        ...

    @abstractmethod
    def __truediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> UniverseMapping[float]:
        """Return the elementwise quotient of two mappings."""
        ...

    @abstractmethod
    def __rtruediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> UniverseMapping[float]:
        """Return the elementwise quotient with reversed operands."""
        ...

    @abstractmethod
    def floor(self) -> UniverseMapping[int]:
        """Return a mapping with values floored to integers."""
        ...

    @abstractmethod
    def truncate(self) -> UniverseMapping[int]:
        """Return a mapping with values truncated to integers."""
        ...


def make_universe_mapping[T: (int, float)](
    m: Mapping[str, T],
    universe: Iterable[str],
    constructor_backend: str = "polars",
) -> UniverseMapping[T]:
    """Create a universe-aligned mapping from an arbitrary mapping.

    Values are re-ordered to match ``universe`` and missing keys are filled with
    zeros. Keeping a stable universe ordering enables faster vectorized
    operations when combining mappings.

    Args:
        m: Mapping of security identifiers to values.
        universe: Ordered universe defining the mapping alignment.
        constructor_backend: Backend used for the concrete mapping type.

    Returns:
        UniverseMapping aligned to ``universe``.
    """
    if not isinstance(m, UniverseMapping) and isinstance(m, Mapping):
        backend_mapping_type = _get_mapping_type_from_backend(constructor_backend)
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
