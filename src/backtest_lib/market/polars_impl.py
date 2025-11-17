from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    Sequence,
    SupportsFloat,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import polars as pl
import polars.datatypes
from numpy.typing import NDArray

from backtest_lib.market.timeseries import Timeseries
from backtest_lib.universe import SecurityName
from backtest_lib.universe.vector_mapping import VectorMapping

if TYPE_CHECKING:
    import pandas as pd

    from backtest_lib.market import SecurityMappings
    from backtest_lib.universe import Universe
    from backtest_lib.universe.vector_ops import Scalar, VectorOps

logger = logging.getLogger(__name__)

_RHS_HANDOFF = object()

Numeric = TypeVar("Numeric", int, float)


POLARS_TO_PYTHON: dict[pl.DataType | type[Any], type[Any]] = {
    pl.Boolean: bool,
    pl.Int8: int,
    pl.Int16: int,
    pl.Int32: int,
    pl.Int64: int,
    pl.UInt8: int,
    pl.UInt16: int,
    pl.UInt32: int,
    pl.UInt64: int,
    pl.Float32: float,
    pl.Float64: float,
    pl.String: str,
    pl.Categorical: str,  # categorical values are stored as strings
    pl.Enum: str,
    pl.Date: dt.date,
    pl.Datetime: dt.datetime,
    pl.Time: dt.time,
    pl.Duration: dt.timedelta,
    pl.Decimal: Decimal,
    pl.Binary: bytes,
    pl.Object: object,
    pl.Null: type(None),
}


class Array1DDTView(Sequence[np.datetime64]):
    """
    Zero-copy 1-D Sequence view over an NDArray[np.datetime64].
    Ensures 1-D at construction; slicing returns another view.

    This serves mainly as a wrapper around NDArray that implements
    Sequence.
    """

    def __init__(self, a: NDArray[np.datetime64]):
        if a.ndim != 1:
            a = a.reshape(-1)  # view when possible
        self._a: NDArray[np.datetime64] = a

    @property
    def array(self) -> NDArray[np.datetime64]:
        return self._a

    def __len__(self) -> int:
        return self._a.shape[0]

    @overload
    def __getitem__(self, index: int) -> np.datetime64: ...
    @overload
    def __getitem__(self, index: slice) -> "Array1DDTView": ...

    def __getitem__(self, index: int | slice) -> np.datetime64 | "Array1DDTView":
        if isinstance(index, slice):
            return Array1DDTView(self._a[index])
        if isinstance(index, (int, np.integer)):
            return self._a[index]
        raise TypeError(f"Invalid index type: {type(index)!r}")

    def __iter__(self) -> Iterator[np.datetime64]:
        for i in range(self._a.shape[0]):
            yield self._a[i]

    def __repr__(self) -> str:
        return f"Array1DDTView({self._a!r})"

    def __array__(self) -> NDArray[np.datetime64]:
        return self._a.astype("datetime64[us]")


@dataclass(frozen=True)
class Axis:
    names: tuple[str, ...]
    pos: dict[str, int]  # name -> index (0..N-1)

    @staticmethod
    def from_names(names: Sequence[SecurityName]) -> Axis:
        names_t = tuple(names)
        return Axis(names_t, {s: i for i, s in enumerate(names_t)})

    def __len__(self):
        return len(self.names)


def _to_npdt64(x: np.datetime64 | str) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[us]")
    if isinstance(x, str):
        return np.datetime64(x, "us")
    raise TypeError(f"Unsupported type {type(x)}")


@dataclass(frozen=True)
class PeriodAxis:
    dt64: NDArray[np.datetime64]
    labels: tuple[str, ...]
    pos: dict[str, int]  # label -> index

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def from_series(date_s: pl.Series, fmt: str = "%Y-%m-%d") -> PeriodAxis:
        if date_s.dtype not in (pl.Date, pl.Datetime):
            date_s = date_s.cast(pl.Datetime("us"))
        labels = tuple(date_s.dt.strftime(fmt).to_list())
        dt64 = date_s.to_numpy(allow_copy=False).astype("datetime64[us]", copy=False)
        # assert np.all(dt64[1:] >= dt64[:-1])
        return PeriodAxis(dt64, labels, {lbl: i for i, lbl in enumerate(labels)})

    def take(self, idxs: Sequence[int] | NDArray[np.int64]) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a sequence of
        integer indices contained in the period axis.
        """
        new_labels = tuple(self.labels[i] for i in idxs)
        return PeriodAxis(
            dt64=self.dt64[idxs],
            labels=tuple(new_labels),
            pos={lbl: i for i, lbl in enumerate(new_labels)},
        )

    def _slice(self, key: slice) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a slice
        of the current PeriodAxis.
        """
        start, stop, step = key.indices(len(self))
        if step == 1:
            return self.slice_contiguous(start, stop)
        period_idxs = np.arange(start, stop, step, dtype=np.int64)
        return self.take(period_idxs)

    def slice_contiguous(self, start: int, stop: int) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a slice
        of the current PeriodAxis.
        """
        new_dt64 = self.dt64[start:stop]
        new_labels = self.labels[start:stop]
        new_pos = {lbl: i for i, lbl in enumerate(new_labels)}
        return PeriodAxis(new_dt64, new_labels, new_pos)

    def bounds_after(
        self, start: np.datetime64, *, inclusive: bool = True
    ) -> tuple[int, int]:
        left = int(
            np.searchsorted(self.dt64, start, side="left" if inclusive else "right")
        )
        return left, len(self.dt64)

    def bounds_before(
        self, end: np.datetime64, *, inclusive: bool = False
    ) -> tuple[int, int]:
        right = int(
            np.searchsorted(self.dt64, end, side="right" if inclusive else "left")
        )
        return 0, right

    def bounds_between(
        self,
        start: np.datetime64,
        end: np.datetime64,
        *,
        closed: str = "both",
    ) -> tuple[int, int]:
        inc_start = closed in ("both", "left")
        inc_end = closed in ("both", "right")
        left, _ = self.bounds_after(start, inclusive=inc_start)
        _, right = self.bounds_before(end, inclusive=inc_end)
        return left, right

    def after(
        self, start: np.datetime64, *, inclusive: bool = True
    ) -> NDArray[np.int64]:
        left, right = self.bounds_after(start, inclusive=inclusive)
        return np.arange(left, right, dtype=np.int64)

    def before(
        self, end: np.datetime64, *, inclusive: bool = False
    ) -> NDArray[np.int64]:
        left, right = self.bounds_before(end, inclusive=inclusive)
        return np.arange(left, right, dtype=np.int64)

    def between(
        self,
        start: np.datetime64,
        end: np.datetime64,
        *,
        closed: str = "left",
    ) -> NDArray[np.int64]:
        left, right = self.bounds_between(start, end, closed=closed)
        return np.arange(left, right, dtype=np.int64)


@dataclass(frozen=True, init=False)
class PolarsTimeseries[T: int | float](Timeseries[T, np.datetime64]):
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
                self._vec[key], self._axis._slice(key), self._name, self._scalar_type
            )

    def before(self, end: np.datetime64 | str, *, inclusive=False) -> Self:
        end = _to_npdt64(end)
        left, right = self._axis.bounds_before(end, inclusive=inclusive)
        return PolarsTimeseries(
            self._vec[left:right],
            self._axis.slice_contiguous(left, right),
            self._name,
            self._scalar_type,
        )

    def after(self, start: np.datetime64 | str, *, inclusive=True) -> Self:
        start = _to_npdt64(start)
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
        start = _to_npdt64(start)
        end = _to_npdt64(end)
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

    def _rhs(self, other: VectorOps[Scalar] | Scalar) -> pl.Series | T:
        if isinstance(other, (int, float)):
            return self._scalar_type(other)
        if isinstance(other, PolarsTimeseries):
            if other._axis is self._axis or other._axis.labels == self._axis.labels:
                return other._vec
            raise ValueError("Axis mismatch: operations require identical PeriodAxis.")
        raise TypeError("Only scalars or same-axis PolarsTimeseries are supported.")

    def __add__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec + rhs, self._axis, self._name, self._scalar_type
        )

    def __radd__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs + self._vec, self._axis, self._name, self._scalar_type
        )

    def __sub__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec - rhs, self._axis, self._name, self._scalar_type
        )

    def __rsub__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs - self._vec, self._axis, self._name, self._scalar_type
        )

    def __mul__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec * rhs, self._axis, self._name, self._scalar_type
        )

    def __rmul__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs * self._vec, self._axis, self._name, self._scalar_type
        )

    def __truediv__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec / rhs, self._axis, self._name, self._scalar_type
        )

    def __rtruediv__(self, other: VectorOps[Scalar] | Scalar) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs / self._vec, self._axis, self._name, self._scalar_type
        )

    def sum(self) -> T:
        return self._scalar_type(self._vec.sum())

    def mean(self) -> T:
        return self._scalar_type(self._vec.mean())

    def abs(self) -> Self:
        return PolarsTimeseries(
            self._vec.abs(), self._axis, self._name, self._scalar_type
        )

    def floor(self) -> PolarsTimeseries[int]:
        return PolarsTimeseries(
            _vec=self._vec.floor(),
            _axis=self._axis,
            _name=self._name,
            _scalar_type=int,
        )


@dataclass(frozen=True, init=False)
class SeriesUniverseMapping[T: (float, int)](VectorMapping[SecurityName, T]):
    names: Universe
    _data: pl.Series
    pos: dict[SecurityName, int] = field(repr=False)
    _scalar_type: type[T]

    @staticmethod
    def from_names_and_data(
        names: Universe,
        data: pl.Series,
        dtype: type[int] | type[float] | None = None,
    ) -> SeriesUniverseMapping:
        return SeriesUniverseMapping(
            names=names,
            _data=data,
            pos={name: i for i, name in enumerate(names)},
            _scalar_type=dtype,
        )

    def __init__(
        self,
        names: Universe,
        _data: pl.Series,
        pos: dict[SecurityName, int],
        _scalar_type: type[int] | type[float] | None = None,
    ):
        n = len(names)
        if len(_data) != n or len(pos) != n:
            raise ValueError(
                f"Row mapping misaligned: names={n}, _data={len(_data)}, pos={len(pos)}"
            )
        if _scalar_type is None:
            _scalar_type = POLARS_TO_PYTHON[_data.dtype]
        if _scalar_type is float and _data.dtype != pl.Float64:
            _data = _data.cast(pl.Float64)
        elif _scalar_type is int and _data.dtype != pl.Int64:
            _data = _data.cast(pl.Int64)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "_data", _data)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "_scalar_type", _scalar_type)

    def as_series(self) -> pl.Series:
        return self._data

    @overload
    def __getitem__(self, key: SecurityName) -> T: ...

    @overload
    def __getitem__(self, key: Iterable[SecurityName]) -> pl.Series: ...

    def __getitem__(self, key: SecurityName | Iterable[SecurityName]) -> T | pl.Series:
        if isinstance(key, SecurityName):
            return self._scalar_type(self._data.item(self.pos[key]))
        elif isinstance(key, list):
            idx = np.fromiter(
                (self.pos[k] for k in key), dtype=np.int64, count=len(key)
            )
            return self._data.gather(idx)
        else:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}")

    def __iter__(self) -> Iterator[SecurityName]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def _rhs(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> tuple[pl.Series | float, type[Any] | None]:
        if isinstance(other, SupportsFloat):
            if not isinstance(other, int) and self._scalar_type is int:
                return float(other), type(other)
            return float(other), self._scalar_type
        elif isinstance(other, SeriesUniverseMapping):
            data = other._data
            if other.names != self.names:
                if not all(name in self.names for name in other.names):
                    logger.debug(f"lhs size: {len(self)}, rhs size: {len(other)}")
                    logger.debug(
                        f"{len([name for name in self.names if name not in other.names])} items found in lhs not in rhs"
                    )
                    return cast(float, _RHS_HANDOFF), None
                data = _mapping_to_series(self, other)
            if other._scalar_type is not int:
                return data, float
            return data, self._scalar_type
        elif isinstance(other, Mapping):
            aligned_series = _mapping_to_series(self, other)
            if (
                any(not isinstance(x, int) for x in other.values())
                and self._scalar_type is int
            ):
                return aligned_series, float
            return aligned_series, self._scalar_type

        raise TypeError(
            "Unsupported operand: only scalars, mappings or SeriesUniverseMapping with compatible axes are allowed."
        )

    def __add__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            return self.__radd__(other)
        return SeriesUniverseMapping(self.names, self._data + rhs, self.pos, new_type)

    def __radd__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs + self._data, self.pos, new_type)

    def __sub__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rsub__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping(self.names, self._data - rhs, self.pos, new_type)

    def __rsub__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs - self._data, self.pos, new_type)

    def __mul__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rmul__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping(self.names, self._data * rhs, self.pos, new_type)

    def __rmul__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs * self._data, self.pos, new_type)

    def __truediv__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        rhs, _ = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rtruediv__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping(self.names, self._data / rhs, self.pos, float)

    def __rtruediv__(
        self, other: VectorOps[T] | SupportsFloat | Mapping
    ) -> SeriesUniverseMapping:
        lhs, _ = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs / self._data, self.pos, float)

    def sum(self) -> T:
        return self._scalar_type(self._data.sum())

    def mean(self) -> T:
        # assert self._data.dtype == pl.Float64
        dt = self._data.dtype
        if not (dt.is_numeric()):
            raise TypeError(f"mean() only supported on numeric dtypes, got {dt!r}")
        m = self._data.mean()
        if m is None:
            raise ValueError("Mean of empty series")
        return self._scalar_type(m)

    def abs(self) -> Self:
        return SeriesUniverseMapping(
            self.names, self._data.abs(), self.pos, self._scalar_type
        )

    def floor(self) -> SeriesUniverseMapping:
        return SeriesUniverseMapping(
            names=self.names, _data=self._data.floor(), pos=self.pos, _scalar_type=int
        )

    @classmethod
    def from_vectors(
        cls, keys: Sequence[str], values: Sequence[float]
    ) -> SeriesUniverseMapping:
        if not isinstance(keys, tuple):
            keys_tuple = tuple(keys)
        else:
            keys_tuple = keys
        keys_tuple = cast(tuple[str, ...], keys_tuple)

        if not isinstance(values, pl.Series):
            values_series = pl.Series(values, dtype=pl.Float64)
        else:
            values_series = values
        if values_series.dtype != pl.Float64:
            values_series = values_series.cast(pl.Float64)

        return SeriesUniverseMapping.from_names_and_data(keys_tuple, values_series)


@dataclass(frozen=True)
class PolarsByPeriod:
    _period_column_df: pl.DataFrame
    _security_column_df: pl.DataFrame = field(repr=False)
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    _period_slice_start: int = field(repr=False, default=0)
    _period_slice_len: int | None = field(repr=False, default=None)  # None => to end

    _row_indexer: np.ndarray | None = field(repr=False, default=None)

    _col_names_cache: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_col_names_cache", tuple(self._period_column_df.columns)
        )

    def __len__(self) -> int:
        total = len(self._col_names_cache)
        if self._period_slice_len is None:
            return max(0, total - self._period_slice_start)
        return max(0, min(self._period_slice_len, total - self._period_slice_start))

    def _abs_col_index(self, logical_i: int) -> int:
        n = len(self)
        i = logical_i if logical_i >= 0 else n + logical_i
        if i < 0 or i >= n:
            raise IndexError(i)
        return self._period_slice_start + i

    def as_df(self, *, show_securities: bool = False) -> pl.DataFrame:
        start = self._period_slice_start
        stop = self._period_slice_start + len(self)
        df = self._period_column_df[:, start:stop]
        if self._row_indexer is not None:
            df = df.select(pl.all().gather(self._row_indexer))
        if show_securities:
            securities_series = pl.Series(self._security_axis.names)
            return df.with_columns(security=securities_series).select(
                "security", pl.all().exclude("security")
            )
        return df

    @overload
    def __getitem__(self, key: SupportsIndex) -> SeriesUniverseMapping: ...
    @overload
    def __getitem__(self, key: slice) -> PolarsPastView: ...

    def __getitem__(self, key: SupportsIndex | slice):
        if isinstance(key, SupportsIndex):
            abs_j = self._abs_col_index(int(key))
            col_name = self._col_names_cache[abs_j]
            s = self._period_column_df.get_column(col_name)
            if self._row_indexer is not None:
                s = s.gather(self._row_indexer)
            scalar_type = POLARS_TO_PYTHON[self.as_df().dtypes[0]]
            return SeriesUniverseMapping(
                names=self._security_axis.names,
                _data=s,
                pos=self._security_axis.pos,
                _scalar_type=scalar_type,
            )

        start, stop, step = key.indices(len(self))
        if step == 1:
            abs_start = self._period_slice_start + start
            abs_stop = self._period_slice_start + stop

            by_period_view = PolarsByPeriod(
                self._period_column_df,
                self._security_column_df,
                self._security_axis,
                self._period_axis,
                _period_slice_start=abs_start,
                _period_slice_len=abs_stop - abs_start,
                _row_indexer=self._row_indexer,
            )

            new_period_cols = self._col_names_cache[abs_start:abs_stop]
            new_period_axis = PeriodAxis(
                dt64=self._period_axis.dt64[abs_start:abs_stop],
                labels=tuple(new_period_cols),
                pos={lbl: i for i, lbl in enumerate(new_period_cols)},
            )

            by_security_view = PolarsBySecurity(
                _security_column_df=self._security_column_df,
                _period_column_df=self._period_column_df,
                _security_axis=self._security_axis,
                _period_axis=new_period_axis,
                _sel_names=self._security_axis.names,
                _period_slice_start=abs_start,
                _period_slice_len=abs_stop - abs_start,
            )

            return PolarsPastView(
                by_period=by_period_view,
                by_security=by_security_view,
                _period_axis=new_period_axis,
                _security_axis=self._security_axis,
            )

        abs_start = self._period_slice_start + start
        abs_stop = self._period_slice_start + stop
        idx = np.arange(abs_start, abs_stop, step, dtype=np.int64)

        period_cols = tuple(self._col_names_cache[i] for i in idx.tolist())
        period_df = self._period_column_df.select(list(period_cols))
        if self._row_indexer is not None:
            period_df = period_df.select(pl.all().gather(self._row_indexer))

        sec_df = self._security_column_df.select(pl.all().gather(idx))
        new_pax = PeriodAxis(
            dt64=self._period_axis.dt64[idx],
            labels=period_cols,
            pos={lbl: i for i, lbl in enumerate(period_cols)},
        )

        return PolarsPastView(
            by_period=PolarsByPeriod(period_df, sec_df, self._security_axis, new_pax),
            by_security=PolarsBySecurity(
                sec_df, period_df, self._security_axis, new_pax
            ),
            _period_axis=new_pax,
            _security_axis=self._security_axis,
        )

    def __iter__(self) -> Iterator[np.datetime64]:
        for period in self._period_axis.dt64:
            yield period

    @overload
    def to_dataframe(self) -> pl.DataFrame: ...

    @overload
    def to_dataframe(self, show_securities: bool) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self, show_securities: bool, backend: Literal["polars"]
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self, show_securities: bool, backend: Literal["pandas"]
    ) -> pd.DataFrame: ...

    def to_dataframe(
        self,
        show_securities: bool = False,
        backend: Literal["polars"] | Literal["pandas"] = "polars",
    ) -> pl.DataFrame | pd.DataFrame:
        if backend == "polars":
            return self.as_df(show_securities=show_securities)
        elif backend == "pandas":
            return self.as_df(show_securities=show_securities).to_pandas()
        raise ValueError(f"'{backend}' is not a valid DataFrame backend.")


@dataclass(frozen=True)
class PolarsBySecurity:
    _security_column_df: pl.DataFrame
    _period_column_df: pl.DataFrame = field(repr=False)
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    _period_slice_start: int = field(repr=False, default=0)
    _period_slice_len: int | None = field(repr=False, default=None)
    _sel_names: tuple[str, ...] | None = field(repr=False, default=None)

    def __len__(self) -> int:
        return len(self._security_axis)

    def as_df(self, *, show_periods: bool = True) -> pl.DataFrame:
        if self._sel_names is None:
            df = self._security_column_df
        else:
            df = self._security_column_df.select(list(self._sel_names))
        if self._period_slice_start != 0 or self._period_slice_len is not None:
            df = df.slice(self._period_slice_start, self._period_slice_len)
        if show_periods:
            periods_series = pl.Series(self._period_axis.dt64, dtype=pl.Datetime)
            return df.with_columns(date=periods_series).select(
                "date", pl.all().exclude("date")
            )
        return df

    @overload
    def __getitem__(self, key: str) -> PolarsTimeseries: ...
    @overload
    def __getitem__(self, key: Iterable[str]) -> PolarsPastView: ...

    def __getitem__(
        self, key: str | Iterable[str]
    ) -> PolarsTimeseries | PolarsPastView:
        if isinstance(key, SecurityName):
            if self._sel_names is not None and key not in self._sel_names:
                raise KeyError(key)

            s = self._security_column_df.get_column(key)
            if self._period_slice_start != 0 or self._period_slice_len is not None:
                s = s.slice(self._period_slice_start, self._period_slice_len)

            start = self._period_slice_start
            stop = start + (
                len(self._period_axis.labels) - start
                if self._period_slice_len is None
                else self._period_slice_len
            )
            pax = PeriodAxis(
                dt64=self._period_axis.dt64[start:stop],
                labels=tuple(self._period_axis.labels[start:stop]),
                pos={
                    lbl: i for i, lbl in enumerate(self._period_axis.labels[start:stop])
                },
            )
            return PolarsTimeseries(s, pax, key, float)

        names = tuple(key)
        idx = np.fromiter(
            (self._security_axis.pos[n] for n in names),
            dtype=np.int64,
            count=len(names),
        )

        new_security_axis = Axis.from_names(names)

        start = self._period_slice_start
        stop = start + (
            len(self._period_axis.labels) - start
            if self._period_slice_len is None
            else self._period_slice_len
        )
        pax = PeriodAxis(
            dt64=self._period_axis.dt64[start:stop],
            labels=tuple(self._period_axis.labels[start:stop]),
            pos={lbl: i for i, lbl in enumerate(self._period_axis.labels[start:stop])},
        )

        by_security_view = PolarsBySecurity(
            _security_column_df=self._security_column_df,
            _period_column_df=self._period_column_df,
            _security_axis=new_security_axis,
            _period_axis=pax,
            _period_slice_start=self._period_slice_start,
            _period_slice_len=self._period_slice_len,
            _sel_names=names,
        )

        by_period_view = PolarsByPeriod(
            _period_column_df=self._period_column_df,
            _security_column_df=self._security_column_df,
            _security_axis=new_security_axis,
            _period_axis=pax,
            _period_slice_start=self._period_slice_start,
            _period_slice_len=self._period_slice_len,
            _row_indexer=idx,
        )

        return PolarsPastView(
            by_period=by_period_view,
            by_security=by_security_view,
            _period_axis=pax,
            _security_axis=new_security_axis,
        )

    def __iter__(self) -> Iterator[str]:
        for sec in self._security_axis.names:
            yield sec

    @overload
    def to_dataframe(self) -> pl.DataFrame: ...

    @overload
    def to_dataframe(self, show_periods: bool) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self, show_periods: bool, backend: Literal["polars"]
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self, show_periods: bool, backend: Literal["pandas"]
    ) -> pd.DataFrame: ...

    def to_dataframe(
        self,
        show_periods: bool = True,
        backend: Literal["polars"] | Literal["pandas"] = "polars",
    ) -> pl.DataFrame | pd.DataFrame:
        if backend == "polars":
            return self.as_df(show_periods=show_periods)
        elif backend == "pandas":
            return self.as_df(show_periods=show_periods).to_pandas()
        raise ValueError(f"'{backend}' is not a valid DataFrame backend.")


@dataclass(frozen=True)
class PolarsPastView:
    by_period: PolarsByPeriod = field(repr=False)
    by_security: PolarsBySecurity
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    @property
    def periods(self) -> Sequence[np.datetime64]:
        return Array1DDTView(self._period_axis.dt64)

    @property
    def securities(self) -> tuple[SecurityName, ...]:
        return self._security_axis.names

    @staticmethod
    def from_security_mappings(
        ms: SecurityMappings[Any],
        periods: Sequence[np.datetime64],
    ) -> Self:
        if not ms or any(not m for m in ms):
            raise ValueError("Cannot create a PolarsPastView from an empty mapping.")
        if not len(periods) == len(ms):
            raise ValueError(
                "Length of period sequence must match length of security mapping list"
            )

        first_keys = set(ms[0].keys())
        if not all(len(set(m.keys()) ^ first_keys) == 0 for m in ms):
            differing_keys = next(
                (periods[i], set(m.keys()).symmetric_difference(set(first_keys)))
                for i, m in enumerate([dict(x) for x in ms])
                if m.keys() != first_keys
            )
            raise KeyError(
                "All security mappings must have the same keys to create a PolarsPastView.\n"
                f"Found differing keys from first keys (period, keys): {differing_keys}"
            )
        allowed_types: list[type[Any] | pl.DataType] = [float, int]
        allowed_types.extend(
            [k for k, v in POLARS_TO_PYTHON.items() if v in allowed_types]
        )

        unique_passed_types = {type(v) for m in ms for v in m.values()}
        passed_type = next(iter(unique_passed_types), None)
        if not all(x is passed_type for x in unique_passed_types):
            raise ValueError(
                f"All values of the mapping must be the same to create a PolarsPastView, {len(unique_passed_types)} types were passed ({unique_passed_types})"
            )
        if passed_type not in allowed_types:
            raise ValueError(f"Cannot create PolarsPastView of type {passed_type}.")

        periods_series = (
            periods
            if isinstance(periods, pl.Series)
            else pl.Series(
                "date",
                np.asarray(periods).astype("datetime64[us]"),
                dtype=pl.Datetime("us"),
            )
        )
        df = pl.DataFrame({k: [m[k] for m in ms] for k in first_keys}).with_columns(
            date=periods_series
        )

        return PolarsPastView.from_dataframe(df)

    @staticmethod
    def from_dataframe(df: pl.DataFrame | pd.DataFrame) -> Self:
        if not isinstance(df, pl.DataFrame):
            try:
                df = pl.DataFrame(df)
            except Exception as e:
                raise ValueError(
                    f"Cannot create PolarsPastView from '{df.__name__}'. It must be able to be turned into a polars DataFrame with a 'date' column and a column for each security: {e}"
                )
        try:
            dates = df.get_column("date")
        except Exception as e:
            raise ValueError(
                "Input dataframe must have column for 'date' and a column for each security"
            ) from e

        if dates.dtype not in (pl.Date, pl.Datetime):
            dates = dates.cast(pl.Datetime("ms"))

        period_names = dates.dt.to_string()
        non_date_cols = [x for x in df.columns if x != "date"]

        security_column_df = df.select(non_date_cols)
        period_column_df = security_column_df.transpose(column_names=period_names)

        security_axis = Axis.from_names(security_column_df.columns)
        period_axis = PeriodAxis.from_series(dates)

        return PolarsPastView(
            PolarsByPeriod(
                period_column_df, security_column_df, security_axis, period_axis
            ),
            PolarsBySecurity(
                security_column_df, period_column_df, security_axis, period_axis
            ),
            security_axis,
            period_axis,
        )

    def _slice_period(self, left: int, right: int) -> PolarsPastView:
        cols = self.by_period._period_column_df.columns[left:right]
        new_period_df = self.by_period._period_column_df.select(cols)

        win_len = right - left
        new_security_df = self.by_security._security_column_df.select(
            pl.all().slice(left, win_len)
        )

        new_period_axis = self._period_axis.slice_contiguous(left, right)

        return PolarsPastView(
            PolarsByPeriod(
                new_period_df, new_security_df, self._security_axis, new_period_axis
            ),
            PolarsBySecurity(
                new_security_df, new_period_df, self._security_axis, new_period_axis
            ),
            self._security_axis,
            new_period_axis,
        )

    def after(
        self, start: np.datetime64 | str, *, inclusive: bool = True
    ) -> PolarsPastView:
        left, right = self._period_axis.bounds_after(
            _to_npdt64(start), inclusive=inclusive
        )
        return self._slice_period(left, right)

    def before(
        self, end: np.datetime64 | str, *, inclusive: bool = False
    ) -> PolarsPastView:
        left, right = self._period_axis.bounds_before(
            _to_npdt64(end), inclusive=inclusive
        )
        return self._slice_period(left, right)

    def between(
        self,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        closed: str = "left",
    ) -> PolarsPastView:
        left, right = self._period_axis.bounds_between(
            _to_npdt64(start), _to_npdt64(end), closed=closed
        )
        return self._slice_period(left, right)


def _mapping_to_series[T: (float, int)](
    mapping: SeriesUniverseMapping,
    other_mapping: Mapping[Any, T] | VectorMapping[Any, T],
) -> pl.Series:
    keys_touched = len(other_mapping)
    idxs = np.fromiter(
        (mapping.pos[k] for k in other_mapping.keys()),
        dtype=np.int64,
        count=keys_touched,
    )
    vals = np.fromiter(
        (float(v) for v in other_mapping.values()), dtype=float, count=keys_touched
    )
    series = pl.zeros(len(mapping.names), eager=True)
    # TODO: I don't like having to use this function.
    series.scatter(idxs, vals)
    return series
