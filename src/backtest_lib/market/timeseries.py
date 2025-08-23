from __future__ import annotations

from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
    Self,
    overload,
    Iterator,
)


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...


T = TypeVar("T", covariant=True)  # Scalar elements of the vector
Index = TypeVar("Index", bound=Comparable, contravariant=True)  # Type used for indexing


@runtime_checkable
class Timeseries(Protocol[T, Index]):
    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> (
        Self
    ): ...  # can clone, must provide exact items in the index or integer indices

    def between(
        self,
        start: Index | str,
        end: Index | str,
    ) -> Self: ...  # will not clone data, must be contiguous, performs a binary search

    def after(
        self,
        start: Index | str,
        *,
        inclusive: bool = True,  # common expectation: include the start tick
    ) -> Self: ...

    def before(
        self,
        end: Index | str,
        *,
        inclusive: bool = False,  # common expectation: half-open [.., end)
    ) -> Self: ...

    def __iter__(self) -> Iterator[T]: ...

    def __len__(self) -> int: ...
