from __future__ import annotations

from typing import (
    Generic,
    TypeVar,
    runtime_checkable,
    Self,
    overload,
    Iterator,
    Protocol,
)
from backtest_lib.universe.vector_ops import VectorOps, Scalar
from abc import ABC, abstractmethod
import numpy as np

BoolLike = bool | np.bool | np.bool_


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Self) -> BoolLike: ...
    def __le__(self, other: Self) -> BoolLike: ...
    def __gt__(self, other: Self) -> BoolLike: ...
    def __ge__(self, other: Self) -> BoolLike: ...


Index = TypeVar("Index", bound=Comparable, contravariant=True)  # Type used for indexing


class Timeseries(VectorOps[Scalar], Generic[Scalar, Index], ABC):
    @overload
    def __getitem__(self, key: int) -> Scalar: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> (
        Self
    ): ...  # can clone, must provide exact items in the index or integer indices

    @abstractmethod
    def __getitem__(
        self, key: int | slice
    ) -> (
        Scalar | Self
    ): ...  # can clone, must provide exact items in the index or integer indices

    @abstractmethod
    def between(
        self,
        start: Index | str,
        end: Index | str,
    ) -> Self: ...  # will not clone data, must be contiguous, performs a binary search

    @abstractmethod
    def after(
        self,
        start: Index | str,
        *,
        inclusive: bool = True,  # common expectation: include the start tick
    ) -> Self: ...

    @abstractmethod
    def before(
        self,
        end: Index | str,
        *,
        inclusive: bool = False,  # common expectation: half-open [.., end)
    ) -> Self: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Scalar]: ...

    @abstractmethod
    def __len__(self) -> int: ...
