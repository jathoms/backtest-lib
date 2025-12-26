from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Self, TypeVar

from backtest_lib.universe.vector_ops import VectorOps

K_contra = TypeVar(
    "K_contra",
    bound=str,
)


class VectorMapping[K, V: (float, int)](VectorOps[V], Mapping[K, V], ABC):
    @abstractmethod
    def __truediv__(
        self, other: VectorOps | float | int
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorOps | float | int
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def floor(self) -> VectorMapping[K, int]: ...

    @abstractmethod
    def __getitem__(self, key: K) -> V: ...

    @abstractmethod
    def __iter__(self) -> Iterator[K]: ...

    @classmethod
    @abstractmethod
    def from_vectors(cls, keys: Sequence[K_contra], values: Sequence[V]) -> Self: ...
