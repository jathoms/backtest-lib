from __future__ import annotations
from backtest_lib.universe.vector_ops import Scalar
from collections.abc import Mapping
from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import (
    Generic,
    TypeVar,
    SupportsFloat,
    Protocol,
    runtime_checkable,
)
from backtest_lib.universe.vector_ops import VectorOps


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


class VectorMapping(Mapping[K, Scalar], VectorOps[Scalar], Generic[K, Scalar], ABC):
    @abstractmethod
    def __truediv__(
        self, other: VectorOps | SupportsFloat
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorOps | SupportsFloat
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def floor(self) -> VectorMapping[K, int]: ...

    @abstractmethod
    def zeroed(self) -> VectorMapping[K, Scalar]: ...
