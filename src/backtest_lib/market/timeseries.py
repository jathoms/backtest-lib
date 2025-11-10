from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Self,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from backtest_lib.universe.vector_ops import Scalar, VectorOps

BoolLike = bool | np.bool | np.bool_ | NDArray[np.bool_]


@runtime_checkable
class Comparable(Protocol):
    __lt__: Callable[[Any], Any]
    __le__: Callable[[Any], Any]
    __gt__: Callable[[Any], Any]
    __ge__: Callable[[Any], Any]


Index = TypeVar("Index", bound=Comparable, contravariant=True)  # Type used for indexing


class Timeseries(VectorOps[Scalar], Generic[Scalar, Index], ABC):
    @overload
    def __getitem__(self, key: int) -> Scalar: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

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
