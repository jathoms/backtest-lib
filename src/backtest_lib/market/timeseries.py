from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from backtest_lib.market.plotting import TimeseriesPlotAccessor
from backtest_lib.universe.vector_ops import VectorOps

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

BoolLike = bool | np.bool | np.bool_ | NDArray[np.bool_]


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Any, /) -> Any: ...

    def __le__(self, other: Any, /) -> Any: ...

    def __gt__(self, other: Any, /) -> Any: ...

    def __ge__(self, other: Any, /) -> Any: ...


class Timeseries[Scalar: (float, int), Index: Comparable](VectorOps[Scalar], ABC):
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

    @overload
    def to_series(self, backend: Literal["polars"] = "polars") -> pl.Series: ...

    @overload
    def to_series(self, backend: Literal["pandas"]) -> pd.Series: ...

    @abstractmethod
    def to_series(
        self, backend: Literal["polars", "pandas"] = "polars"
    ) -> pl.Series | pd.Series: ...

    @property
    @abstractmethod
    def plot(self) -> TimeseriesPlotAccessor: ...

    @classmethod
    @abstractmethod
    def from_vectors(
        cls,
        values: Iterable[Scalar],
        periods: Iterable[Index],
    ) -> Self: ...
