from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
    TypeVar,
    overload,
)

import numpy as np
import polars as pl

from backtest_lib.market.plotting import (
    TimeseriesPlotAccessor,
)
from backtest_lib.market.polars_impl._axis import PeriodAxis
from backtest_lib.market.polars_impl._helpers import POLARS_TO_PYTHON, to_npdt64
from backtest_lib.market.polars_impl._plotting import PolarsTimeseriesPlotAccessor
from backtest_lib.market.timeseries import Timeseries
from backtest_lib.universe.vector_ops import VectorOps

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


Scalar = TypeVar("Scalar", float, int)
ScalarU = float | int


class AlignmentError(Exception): ...


@dataclass(frozen=True, init=False)
class PolarsTimeseries[T: (float, int)](Timeseries[T, np.datetime64]):
    _vec: pl.Series
    _axis: PeriodAxis
    _name: str
    _scalar_type: type[T]

    def __init__(
        self,
        _vec: pl.Series,
        _axis: PeriodAxis,
        _name: str = "",
        _scalar_type: type[int] | type[float] | None = None,
    ):
        n = len(_axis)
        if len(_vec) != n:
            raise ValueError(f"Timeseries misaligned: axis={n}, vec={len(_vec)}")

        final_name = _name if _name else (_vec.name or "")
        if _vec.name != final_name:
            _vec = _vec.rename(final_name)

        if _scalar_type is None:
            try:
                st = POLARS_TO_PYTHON[_vec.dtype]
            except KeyError as e:
                raise TypeError(
                    f"Unsupported dtype for PolarsTimeseries: {_vec.dtype}"
                ) from e
        else:
            st = _scalar_type

        if st is float and _vec.dtype != pl.Float64:
            _vec = _vec.cast(pl.Float64)
        elif st is int and _vec.dtype != pl.Int64:
            _vec = _vec.cast(pl.Int64)
        elif st is bool and _vec.dtype != pl.Boolean:
            _vec = _vec.cast(pl.Boolean)

        object.__setattr__(self, "_vec", _vec)
        object.__setattr__(self, "_axis", _axis)
        object.__setattr__(self, "_name", final_name)
        object.__setattr__(self, "_scalar_type", st)

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> (
        Self
    ): ...  # can clone, must provide exact items in the index or integer indices

    def __getitem__(self, key: int | slice) -> T | Self:
        if isinstance(key, int):
            val: T = self._scalar_type(self._vec[key])
            return val
            # return self._scalar_type(self._vec[key])
        else:
            return PolarsTimeseries[T](
                self._vec[key], self._axis.slice(key), self._name, self._scalar_type
            )

    def before(self, end: np.datetime64 | str, *, inclusive=False) -> Self:
        end = to_npdt64(end)
        left, right = self._axis.bounds_before(end, inclusive=inclusive)
        return PolarsTimeseries(
            self._vec[left:right],
            self._axis.slice_contiguous(left, right),
            self._name,
            self._scalar_type,
        )

    def after(self, start: np.datetime64 | str, *, inclusive=True) -> Self:
        start = to_npdt64(start)
        left, right = self._axis.bounds_after(start, inclusive=inclusive)
        return PolarsTimeseries(
            self._vec[left:right],
            self._axis.slice_contiguous(left, right),
            self._name,
            self._scalar_type,
        )

    def between(
        self,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        closed: str = "left",
    ) -> Self:
        start = to_npdt64(start)
        end = to_npdt64(end)
        left, right = self._axis.bounds_between(start, end, closed=closed)
        return PolarsTimeseries(
            self._vec[left:right],
            self._axis.slice_contiguous(left, right),
            self._name,
            self._scalar_type,
        )

    def __iter__(self) -> Iterator[T]:
        return iter(self._vec)

    def __len__(self) -> int:
        return len(self._axis)

    @overload
    def to_series(self, backend: Literal["polars"]) -> pl.Series: ...
    @overload
    def to_series(self, backend=...) -> pl.Series: ...
    @overload
    def to_series(self, backend: Literal["pandas"]) -> pd.Series: ...
    def to_series(
        self, backend: Literal["polars", "pandas"] = "polars"
    ) -> pl.Series | pd.Series:
        if backend == "polars":
            return self._vec
        elif backend == "pandas":
            return self._vec.to_pandas()
        else:
            raise ValueError(backend)

    # TODO: add scalar type return in here so that
    # the scalar types are properly kept track of.
    def _rhs(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> tuple[pl.Series | T, type[ScalarU]]:
        if isinstance(other, (int, float)):
            return self._scalar_type(other), self._scalar_type
        if isinstance(other, PolarsTimeseries):
            if other._axis is self._axis or other._axis.labels == self._axis.labels:
                return other._vec, other._scalar_type
            raise ValueError("Axis mismatch: operations require identical PeriodAxis.")
        raise TypeError("Only scalars or same-axis PolarsTimeseries are supported.")

    def __add__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(self._vec + rhs, self._axis, self._name, scalar_type)

    def __radd__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(lhs + self._vec, self._axis, self._name, scalar_type)

    def __sub__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(self._vec - rhs, self._axis, self._name, scalar_type)

    def __rsub__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(lhs - self._vec, self._axis, self._name, scalar_type)

    def __mul__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(self._vec * rhs, self._axis, self._name, scalar_type)

    def __rmul__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs, scalar_type = self._rhs(other)
        return PolarsTimeseries(lhs * self._vec, self._axis, self._name, scalar_type)

    def __truediv__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[float]:
        rhs, scalar_type = self._rhs(other)
        return PolarsTimeseries[float](self._vec / rhs, self._axis, self._name, float)

    def __rtruediv__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[float]:
        lhs, scalar_type = self._rhs(other)
        return PolarsTimeseries[float](lhs / self._vec, self._axis, self._name, float)

    def sum(self) -> T:
        return self._scalar_type(self._vec.sum())

    def mean(self) -> T:
        return self._scalar_type(self._vec.mean())

    def abs(self) -> PolarsTimeseries[T]:
        return PolarsTimeseries[T](
            self._vec.abs(), self._axis, self._name, self._scalar_type
        )

    def floor(self) -> PolarsTimeseries[int]:
        return PolarsTimeseries(
            _vec=self._vec.floor(),
            _axis=self._axis,
            _name=self._name,
            _scalar_type=int,
        )

    def truncate(self) -> PolarsTimeseries[int]:
        return PolarsTimeseries(
            _vec=self._vec.cast(pl.Int64),
            _axis=self._axis,
            _name=self._name,
            _scalar_type=int,
        )

    @property
    def plot(self) -> TimeseriesPlotAccessor:
        return PolarsTimeseriesPlotAccessor(self)

    def from_vectors(
        values: Iterable[Scalar],
        periods: Iterable[np.datetime64],
        name: str = "",
    ) -> PolarsTimeseries[Scalar]:
        values_series = pl.Series(np.asarray(values))
        periods_series = pl.Series(
            np.asarray(periods, dtype="datetime64[us]"), dtype=pl.Datetime("us")
        )

        values_dtype = POLARS_TO_PYTHON[values_series.dtype]

        if values_dtype not in (int, float):
            raise TypeError(
                "Cannot create PolarsTimeseries with passed values. "
                f"Type was determined to be {values_dtype}"
            )

        if len(values_series) != len(periods_series):
            raise AlignmentError(
                "Length of values must match length of periods, "
                f"lengths were {len(values_series)} and "
                "{len(periods_series)} respectively."
            )

        period_axis = PeriodAxis.from_series(periods_series)

        return PolarsTimeseries(
            _vec=values_series, _axis=period_axis, _name=name, _scalar_type=values_dtype
        )
