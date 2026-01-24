"""Vectorized mapping primitives for numeric data.

These mappings support elementwise arithmetic and can be implemented with
backend-specific vectorized operations. Performance is best when two mappings
share the same key order, because backends can apply operations without
reordering. The benchmarks in
``tests/benchmark/polars_impl/test_universe_mapping_benchmark.py`` illustrate the
penalty for mismatched ordering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Self, TypeVar, overload

K_contra = TypeVar(
    "K_contra",
    bound=str,
)
V_co = TypeVar("V_co", int, float, covariant=True)


class VectorMapping[K, V_co](Mapping[K, V_co], ABC):
    """Mapping with vectorized arithmetic semantics.

    Backends may rely on matching key order to perform fast vector operations.
    When key order differs between two mappings, operations remain correct but
    typically require reindexing and incur a performance cost.
    """

    @overload
    def __add__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __add__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __add__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __add__(self, other) -> VectorMapping:
        """Return the elementwise sum of two mappings."""
        ...

    @overload
    def __radd__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __radd__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __radd__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __radd__(self, other) -> VectorMapping:
        """Return the elementwise sum with reversed operands."""
        ...

    @overload
    def __sub__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __sub__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __sub__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __sub__(self, other) -> VectorMapping:
        """Return the elementwise difference of two mappings."""
        ...

    @overload
    def __rsub__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __rsub__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __rsub__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __rsub__(self, other) -> VectorMapping:
        """Return the elementwise difference with reversed operands."""
        ...

    @overload
    def __mul__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __mul__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __mul__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __mul__(self, other) -> VectorMapping:
        """Return the elementwise product of two mappings."""
        ...

    @overload
    def __rmul__(
        self: VectorMapping[K, int],
        other: int | VectorMapping[K, int] | Mapping[K, int],
    ) -> VectorMapping[K, int]: ...
    @overload
    def __rmul__(
        self: VectorMapping[K, int],
        other: float | VectorMapping[K, float] | Mapping[K, float],
    ) -> VectorMapping[K, float]: ...
    @overload
    def __rmul__(
        self: VectorMapping[K, float],
        other: int
        | float
        | VectorMapping[K, int]
        | VectorMapping[K, float]
        | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...
    @abstractmethod
    def __rmul__(self, other) -> VectorMapping:
        """Return the elementwise product with reversed operands."""
        ...

    @abstractmethod
    def __truediv__(
        self,
        other: VectorMapping | float | int | Mapping[K, int | float],
    ) -> VectorMapping[K, float]:
        """Return the elementwise quotient of two mappings."""
        ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorMapping | float | int | Mapping[K, int | float]
    ) -> VectorMapping[K, float]:
        """Return the elementwise quotient with reversed operands."""
        ...

    @abstractmethod
    def sum(self) -> float:
        """Return the sum of all values."""
        ...

    # Default implementations of mean(), override this
    # where possible with faster vector operations
    def mean(self) -> float:
        n = len(self)
        if n == 0:
            raise ValueError(f"mean of empty {type(self)}")
        return self.sum() / n

    @abstractmethod
    def abs(self) -> Self:
        """Return a mapping with absolute values."""
        ...

    @abstractmethod
    def truncate(self) -> VectorMapping[K, int]:
        """Return a mapping with values truncated to integers."""
        ...

    @abstractmethod
    def floor(self) -> VectorMapping[K, int]:
        """Return a mapping with values floored to integers."""
        ...

    @abstractmethod
    def __getitem__(self, key: K) -> V_co:
        """Return the value for a key."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[K]:
        """Return an iterator over keys in order."""
        ...

    @classmethod
    @abstractmethod
    def from_vectors(cls, keys: Iterable[K_contra], values: Iterable[V_co]) -> Self:
        """Create a mapping from ordered key/value vectors."""
        ...
