from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Mapping, Protocol, TypeVar, runtime_checkable

from backtest_lib.universe.vector_ops import VectorOps

# K invariant (to add/sub/div/mul a mapping with another mapping, the key type must be able to be compared directly)
K = TypeVar("K")
M = TypeVar("M", bound="VectorMapping", covariant=True)

K_contra = TypeVar("K_contra", contravariant=True)
Scalar_contra = TypeVar("Scalar_contra", float, int, contravariant=True)


@runtime_checkable
class VectorMappingConstructor(Protocol[M, K_contra, Scalar_contra]):
    @classmethod
    def from_vectors(
        cls, keys: Sequence[K_contra], values: Sequence[Scalar_contra]
    ) -> M: ...


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
