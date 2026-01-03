from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
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

logger = logging.getLogger(__name__)


Scalar = TypeVar("Scalar", float, int)
ScalarU = float | int


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

    def as_series(self) -> pl.Series:
        return self._vec

    # TODO: add scalar type return in here so that
    # the scalar types are properly kept track of.
    def _rhs(self, other: VectorOps[Scalar] | ScalarU) -> pl.Series | T:
        if isinstance(other, (int, float)):
            return self._scalar_type(other)
        if isinstance(other, PolarsTimeseries):
            if other._axis is self._axis or other._axis.labels == self._axis.labels:
                return other._vec
            raise ValueError("Axis mismatch: operations require identical PeriodAxis.")
        raise TypeError("Only scalars or same-axis PolarsTimeseries are supported.")

    def __add__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec + rhs, self._axis, self._name, self._scalar_type
        )

    def __radd__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs + self._vec, self._axis, self._name, self._scalar_type
        )

    def __sub__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec - rhs, self._axis, self._name, self._scalar_type
        )

    def __rsub__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs - self._vec, self._axis, self._name, self._scalar_type
        )

    def __mul__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec * rhs, self._axis, self._name, self._scalar_type
        )

    def __rmul__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[int] | PolarsTimeseries[float]:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs * self._vec, self._axis, self._name, self._scalar_type
        )

    def __truediv__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[float]:
        rhs = self._rhs(other)
        return PolarsTimeseries[float](self._vec / rhs, self._axis, self._name, float)

    def __rtruediv__(
        self, other: VectorOps[Scalar] | ScalarU
    ) -> PolarsTimeseries[float]:
        lhs = self._rhs(other)
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

    @property
    def plot(self) -> TimeseriesPlotAccessor:
        return PolarsTimeseriesPlotAccessor(self)
