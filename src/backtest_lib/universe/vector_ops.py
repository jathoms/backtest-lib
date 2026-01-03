from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Self, TypeVar, overload

_Other_scalar = TypeVar("_Other_scalar", int, float, contravariant=True)


class VectorOps[Scalar: (int, float)](Sized, ABC):
    @overload
    def __add__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __add__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __add__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __add__(self, other: VectorOps | int | float) -> VectorOps: ...

    @overload
    def __radd__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __radd__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __radd__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __radd__(self, other: VectorOps | int | float) -> VectorOps: ...

    @overload
    def __sub__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __sub__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __sub__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __sub__(self, other: VectorOps | int | float) -> VectorOps: ...

    @overload
    def __rsub__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __rsub__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __rsub__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __rsub__(self, other: VectorOps | int | float) -> VectorOps: ...

    @overload
    def __mul__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __mul__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __mul__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __mul__(self, other: VectorOps | int | float) -> VectorOps: ...

    @overload
    def __rmul__(
        self: VectorOps[int],
        other: int | VectorOps[int],
    ) -> VectorOps[int]: ...
    @overload
    def __rmul__(
        self: VectorOps[int],
        other: float | VectorOps[float],
    ) -> VectorOps[float]: ...
    @overload
    def __rmul__(
        self: VectorOps[float],
        other: int | float | VectorOps[int] | VectorOps[float],
    ) -> VectorOps[float]: ...
    @abstractmethod
    def __rmul__(self, other: VectorOps | int | float) -> VectorOps: ...

    @abstractmethod
    def __truediv__(self, other: VectorOps | int | float) -> VectorOps[float]: ...

    @abstractmethod
    def __rtruediv__(self, other: VectorOps | int | float) -> VectorOps[float]: ...

    @abstractmethod
    def sum(self) -> float: ...

    # Default implementations of mean(), override this
    # where possible with faster vector operations
    def mean(self) -> float:
        n = len(self)
        if n == 0:
            raise ValueError(f"mean of empty {type(self)}")
        return self.sum() / n

    def abs(self) -> Self: ...

    @abstractmethod
    def floor(self) -> VectorOps[int]: ...

    @abstractmethod
    def truncate(self) -> VectorOps[int]: ...
