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
    def __add__(self, other) -> VectorMapping: ...

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
    def __radd__(self, other) -> VectorMapping: ...

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
    def __sub__(self, other) -> VectorMapping: ...

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
    def __rsub__(self, other) -> VectorMapping: ...

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
    def __mul__(self, other) -> VectorMapping: ...

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
    def __rmul__(self, other) -> VectorMapping: ...

    @abstractmethod
    def __truediv__(
        self,
        other: VectorMapping | float | int | Mapping[K, int | float],
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def __rtruediv__(
        self, other: VectorMapping | float | int | Mapping[K, int | float]
    ) -> VectorMapping[K, float]: ...

    @abstractmethod
    def sum(self) -> float: ...

    # Default implementations of mean(), override this
    # where possible with faster vector operations
    def mean(self) -> float:
        n = len(self)
        if n == 0:
            raise ValueError(f"mean of empty {type(self)}")
        return self.sum() / n

    @abstractmethod
    def abs(self) -> Self: ...

    @abstractmethod
    def truncate(self) -> VectorMapping[K, int]: ...

    @abstractmethod
    def floor(self) -> VectorMapping[K, int]: ...

    @abstractmethod
    def __getitem__(self, key: K) -> V_co: ...

    @abstractmethod
    def __iter__(self) -> Iterator[K]: ...

    @classmethod
    @abstractmethod
    def from_vectors(cls, keys: Iterable[K_contra], values: Iterable[V_co]) -> Self: ...
