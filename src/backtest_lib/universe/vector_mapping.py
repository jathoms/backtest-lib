from __future__ import annotations

from abc import abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Sequence, ValuesView
from typing import (
    Protocol,
    SupportsFloat,
    TypeVar,
    runtime_checkable,
)

from backtest_lib.universe.vector_ops import Scalar, VectorOps

# K invariant (to add/sub/div/mul a mapping with another mapping, the key type must be able to be compared directly)
K = TypeVar("K")
M = TypeVar("M", bound="VectorMapping", covariant=True)

K_contra = TypeVar("K_contra", contravariant=True)
Scalar_contra = TypeVar("Scalar_contra", bound=SupportsFloat, contravariant=True)


@runtime_checkable
class VectorMappingConstructor(Protocol[M, K_contra, Scalar_contra]):
    @classmethod
    def from_vectors(
        cls, keys: Sequence[K_contra], values: Sequence[Scalar_contra]
    ) -> M: ...


Other_scalar = TypeVar("Other_scalar", int, float)


@runtime_checkable
class VectorMapping(VectorOps[Scalar], Protocol[K, Scalar]):
    @abstractmethod
    def __truediv__(
        self, other: VectorOps | Other_scalar
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorOps | Other_scalar
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def floor(self) -> VectorMapping[K, int]: ...

    @abstractmethod
    def __getitem__(self, key: K) -> Scalar: ...

    def keys(self) -> KeysView[K]:
        "D.keys() -> a set-like object providing a view on D's keys"
        return KeysView(self)

    def items(self) -> ItemsView[K, Scalar]:
        "D.items() -> a set-like object providing a view on D's items"
        return ItemsView(self)

    def values(self) -> ValuesView[Scalar]:
        "D.values() -> an object providing a view on D's values"
        return ValuesView(self)

    def get(self, key, default=None) -> Scalar | None:
        "D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None."
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self) -> Iterator[K]: ...
