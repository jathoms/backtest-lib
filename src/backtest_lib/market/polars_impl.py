from __future__ import annotations
import polars.datatypes

from typing import cast
from collections.abc import Iterator
from backtest_lib.universe.vector_mapping import VectorMapping
from backtest_lib.universe.vector_ops import VectorOps
from dataclasses import dataclass, field
import polars as pl
import numpy as np
from typing import Sequence, SupportsIndex, overload, SupportsFloat
from numpy.typing import NDArray
from backtest_lib.universe import SecurityName, Universe
from typing import Generic, TypeVar
from backtest_lib.market.timeseries import Timeseries

T = TypeVar("T", int, float)


@dataclass(frozen=True)
class Axis:
    names: tuple[str, ...]
    pos: dict[str, int]  # name -> index (0..N-1)

    @staticmethod
    def from_names(names: Sequence[SecurityName]) -> Axis:
        names_t = tuple(names)
        return Axis(names_t, {s: i for i, s in enumerate(names_t)})


def _to_npdt64(x: np.datetime64 | str) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    if isinstance(x, str):
        return np.datetime64(x, "ns")
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
            date_s = date_s.cast(pl.Datetime("ns"))
        labels = tuple(date_s.dt.strftime(fmt).to_list())
        dt64 = date_s.to_numpy(allow_copy=False).astype("datetime64[ns]", copy=False)
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


@dataclass(frozen=True)
class PolarsTimeseries(Timeseries[T, np.datetime64], Generic[T]):
    _vec: NDArray[np.float64]
    _axis: PeriodAxis
    _name: str
    _scalar_type: type[T]

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> (
        PolarsTimeseries
    ): ...  # can clone, must provide exact items in the index or integer indices

    def __getitem__(self, key: int | slice) -> T | PolarsTimeseries:
        if isinstance(key, int):
            return self._scalar_type(self._vec[key])
        else:
            return PolarsTimeseries(
                self._vec[key], self._axis._slice(key), self._name, self._scalar_type
            )

    def before(self, end: np.datetime64 | str, *, inclusive=False) -> PolarsTimeseries:
        end = _to_npdt64(end)
        left, right = self._axis.bounds_before(end, inclusive=inclusive)
        return PolarsTimeseries(
            self._vec[left:right],
            self._axis.slice_contiguous(left, right),
            self._name,
            self._scalar_type,
        )

    def after(self, start: np.datetime64 | str, *, inclusive=True) -> PolarsTimeseries:
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
    ) -> PolarsTimeseries:
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

    def as_array(self) -> NDArray[np.float64]:
        return self._vec

    def as_series(self) -> pl.Series:
        return pl.Series(name=self._name, values=self._vec, dtype=pl.Float64)

    def _rhs(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> NDArray[np.float64] | T:
        if isinstance(other, SupportsFloat):
            return self._scalar_type(float(other))
        if isinstance(other, PolarsTimeseries):
            if other._axis is self._axis or other._axis.labels == self._axis.labels:
                return other._vec
            raise ValueError("Axis mismatch: operations require identical PeriodAxis.")
        raise TypeError("Only scalars or same-axis PolarsTimeseries are supported.")

    def __add__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec + rhs, self._axis, self._name, self._scalar_type
        )

    def __radd__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs + self._vec, self._axis, self._name, self._scalar_type
        )

    def __sub__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec - rhs, self._axis, self._name, self._scalar_type
        )

    def __rsub__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs - self._vec, self._axis, self._name, self._scalar_type
        )

    def __mul__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec * rhs, self._axis, self._name, self._scalar_type
        )

    def __rmul__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs * self._vec, self._axis, self._name, self._scalar_type
        )

    def __truediv__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        rhs = self._rhs(other)
        return PolarsTimeseries(
            self._vec / rhs, self._axis, self._name, self._scalar_type
        )

    def __rtruediv__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> PolarsTimeseries:
        lhs = self._rhs(other)
        return PolarsTimeseries(
            lhs / self._vec, self._axis, self._name, self._scalar_type
        )

    def sum(self) -> T:
        return self._scalar_type(self._vec.sum())

    def mean(self) -> T:
        return self._scalar_type(self._vec.mean())

    def floor(self) -> PolarsTimeseries[int]:
        return PolarsTimeseries(
            _vec=np.floor(self._vec),
            _axis=self._axis,
            _name=self._name,
            _scalar_type=int,
        )


@dataclass(frozen=True)
class SeriesUniverseMapping(VectorMapping[SecurityName, T], Generic[T]):
    names: Universe
    _data: pl.Series
    pos: dict[SecurityName, int] = field(repr=False)
    _scalar_type: type[int] | type[float] = float

    @staticmethod
    def from_names_and_data(
        names: Universe, data: pl.Series, dtype: type[int] | type[float] = float
    ) -> SeriesUniverseMapping:
        return SeriesUniverseMapping(
            names=names,
            _data=data,
            pos={name: i for i, name in enumerate(names)},
            _scalar_type=dtype,
        )

    def as_series(self) -> pl.Series:
        return self._data

    def __post_init__(self):
        n = len(self.names)
        if len(self._data) != n or len(self.pos) != n:
            raise ValueError(
                f"Row mapping misaligned: names={n}, data={len(self._data)}, pos={len(self.pos)}"
            )
        # ensure pos matches the declared order
        if any(self.pos[name] != i for i, name in enumerate(self.names)):
            raise ValueError("pos mapping does not match names order")
        if self._scalar_type is float and self._data.dtype != pl.Float64:
            object.__setattr__(self, "_data", self._data.cast(pl.Float64))
        elif self._scalar_type is int and self._data.dtype != pl.Int64:
            object.__setattr__(self, "_data", self._data.cast(pl.Int64))

    @overload
    def __getitem__(self, key: SecurityName) -> T: ...

    @overload
    def __getitem__(self, key: list[SecurityName]) -> pl.Series: ...

    def __getitem__(self, key: SecurityName | list[SecurityName]) -> T | pl.Series:
        if isinstance(key, SecurityName):
            return cast(T, self._scalar_type(self._data.item(self.pos[key])))
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
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> pl.Series | float:
        if isinstance(other, SupportsFloat):
            return float(other)
        if isinstance(other, SeriesUniverseMapping):
            if other.names != self.names:
                raise ValueError(
                    "Axis mismatch: operations between SeriesUniverseMapping require identical 'names'."
                )
            return other._data
        raise TypeError(
            "Unsupported operand: only scalars or SeriesUniverseMapping with identical axis are allowed."
        )

    def __add__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        rhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, self._data + rhs, self.pos, self._scalar_type
        )

    def __radd__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        lhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, lhs + self._data, self.pos, self._scalar_type
        )

    def __sub__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        rhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, self._data - rhs, self.pos, self._scalar_type
        )

    def __rsub__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        lhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, lhs - self._data, self.pos, self._scalar_type
        )

    def __mul__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        rhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, self._data * rhs, self.pos, self._scalar_type
        )

    def __rmul__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        lhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, lhs * self._data, self.pos, self._scalar_type
        )

    def __truediv__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        rhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, self._data / rhs, self.pos, self._scalar_type
        )

    def __rtruediv__(
        self, other: VectorOps[SupportsFloat] | SupportsFloat
    ) -> SeriesUniverseMapping:
        lhs = self._rhs(other)
        return SeriesUniverseMapping(
            self.names, lhs / self._data, self.pos, self._scalar_type
        )

    def sum(self) -> T:
        return cast(T, self._scalar_type(self._data.sum()))

    def mean(self) -> T:
        # assert self._data.dtype == pl.Float64
        dt = self._data.dtype
        if not (dt.is_numeric()):
            raise TypeError(f"mean() only supported on numeric dtypes, got {dt!r}")
        m = self._data.mean()
        if m is None:
            raise ValueError("mean of empty series")
        mf = float(cast(SupportsFloat, m))

        if self._scalar_type is int:
            return cast(T, int(mf))
        else:
            return cast(T, mf)

    def floor(self) -> SeriesUniverseMapping[int]:
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

    def as_df(self) -> pl.DataFrame:
        return self._period_column_df

    @overload
    def __getitem__(self, key: int) -> SeriesUniverseMapping: ...

    @overload
    def __getitem__(self, key: slice) -> PolarsPastView: ...

    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> SeriesUniverseMapping | PolarsPastView:
        if isinstance(key, SupportsIndex):
            series = self._period_column_df.get_column(
                self._period_column_df.columns[key]
            )
            return SeriesUniverseMapping(
                names=self._security_axis.names,
                _data=series,
                pos=self._security_axis.pos,
                _scalar_type=float,
            )
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._period_axis))

            new_period_cols = self._period_column_df.columns[key]
            new_period_df = self._period_column_df.select(new_period_cols)

            period_idxs = np.arange(start, stop, step, dtype=np.int64)
            new_security_df = self._security_column_df.select(
                pl.all().gather(period_idxs)
            )

            new_period_axis = PeriodAxis(
                dt64=self._period_axis.dt64[period_idxs],
                labels=tuple(new_period_cols),
                pos={lbl: i for i, lbl in enumerate(new_period_cols)},
            )

            return PolarsPastView(
                by_period=(
                    PolarsByPeriod(
                        new_period_df,
                        new_security_df,
                        self._security_axis,
                        new_period_axis,
                    )
                ),
                by_security=(
                    PolarsBySecurity(
                        new_security_df,
                        new_period_df,
                        self._security_axis,
                        new_period_axis,
                    )
                ),
                _period_axis=new_period_axis,
                _security_axis=self._security_axis,
            )
        else:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}")

    def __len__(self) -> int:
        return len(self._period_column_df.columns)


@dataclass(frozen=True)
class PolarsBySecurity:
    _security_column_df: pl.DataFrame
    _period_column_df: pl.DataFrame = field(repr=False)
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    def as_df(self) -> pl.DataFrame:
        return self._security_column_df

    @overload
    def __getitem__(self, key: SecurityName) -> PolarsTimeseries: ...

    @overload
    def __getitem__(self, key: list) -> PolarsPastView: ...

    def __getitem__(
        self, key: SecurityName | list
    ) -> PolarsTimeseries | PolarsPastView:
        if isinstance(key, SecurityName):
            vec = self._security_column_df.get_column(key).to_numpy()
            return PolarsTimeseries(vec, self._period_axis, key, float)

        names = key

        idx = np.fromiter(
            (self._security_axis.pos[n] for n in names),
            dtype=np.int64,
            count=len(names),
        )

        new_security_df = self._security_column_df.select(names)
        new_period_df = self._period_column_df.select(pl.all().gather(idx))

        new_security_axis = Axis.from_names(names)

        return PolarsPastView(
            by_period=PolarsByPeriod(
                new_period_df, new_security_df, new_security_axis, self._period_axis
            ),
            by_security=PolarsBySecurity(
                new_security_df, new_period_df, new_security_axis, self._period_axis
            ),
            _period_axis=self._period_axis,
            _security_axis=new_security_axis,
        )

    def __len__(self) -> int:
        return len(self._security_column_df.columns)


@dataclass(frozen=True)
class PolarsPastView:
    by_period: PolarsByPeriod
    by_security: PolarsBySecurity
    _security_axis: Axis
    _period_axis: PeriodAxis

    @staticmethod
    def from_data_frame(df: pl.DataFrame) -> PolarsPastView:
        try:
            dates = df.get_column("date", default=None)
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
