from __future__ import annotations
from dataclasses import dataclass, field
import polars as pl
import numpy as np
from typing import Sequence, SupportsIndex, overload, Hashable
from collections.abc import Iterator, Mapping
from numpy.typing import NDArray
from backtest_lib.universe import SecurityName, Universe


@dataclass(frozen=True)
class Axis:
    names: tuple[Hashable, ...]
    pos: dict[Hashable, int]  # name -> index (0..N-1)

    @staticmethod
    def from_names(names: Sequence[SecurityName]) -> Axis:
        names_t = tuple(names)
        return Axis(names_t, {s: i for i, s in enumerate(names_t)})


@dataclass(frozen=True)
class PeriodAxis:
    dt64: NDArray[np.datetime64]
    labels: tuple[str, ...]
    pos: dict[str, int]  # label -> index

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def from_series(date_s: pl.Series, fmt: str = "%Y-%m-%d") -> "PeriodAxis":
        if date_s.dtype not in (pl.Date, pl.Datetime):
            date_s = date_s.cast(pl.Datetime("ns"))
        labels = tuple(date_s.dt.strftime(fmt).to_list())
        dt64 = date_s.to_numpy(allow_copy=False).astype("datetime64[ns]", copy=False)
        # assert np.all(dt64[1:] >= dt64[:-1])
        return PeriodAxis(dt64, labels, {lbl: i for i, lbl in enumerate(labels)})

    # convenience for date-range slicing
    def range(
        self, start: np.datetime64, end: np.datetime64, *, closed: str = "both"
    ) -> NDArray[np.int64]:
        dt = self.dt64
        if closed in ("both", "left"):
            left = np.searchsorted(dt, start, side="left")
        else:
            left = np.searchsorted(dt, start, side="right")
        if closed in ("both", "right"):
            right = np.searchsorted(dt, end, side="right")
        else:
            right = np.searchsorted(dt, end, side="left")
        return np.arange(left, right, dtype=np.int64)


@dataclass(frozen=True)
class ArrayRowMapping(Mapping[SecurityName, float]):
    names: Universe
    data: NDArray[np.float64]
    pos: dict[SecurityName, int] = field(repr=False)

    def __post_init__(self):
        if self.data.dtype != np.float64:
            # normalize once so downstream math is predictable
            object.__setattr__(self, "data", self.data.astype(np.float64, copy=False))
        n = len(self.names)
        if len(self.data) != n or len(self.pos) != n:
            raise ValueError(
                f"Row mapping misaligned: names={n}, data={len(self.data)}, pos={len(self.pos)}"
            )
        # ensure pos matches the declared order
        if any(self.pos[name] != i for i, name in enumerate(self.names)):
            raise ValueError("pos mapping does not match names order")
        # expose as read-only
        self.data.setflags(write=False)

    @overload
    def __getitem__(self, key: SecurityName) -> float: ...

    @overload
    def __getitem__(self, key: list[SecurityName]) -> NDArray[np.float64]: ...

    def __getitem__(
        self, key: SecurityName | list[SecurityName]
    ) -> float | NDArray[np.float64]:
        if isinstance(key, SecurityName):
            return float(self.data[self.pos[key]])
        elif isinstance(key, list):
            idx = np.fromiter(
                (self.pos[k] for k in key), dtype=np.int64, count=len(key)
            )
            return self.data[idx]
        else:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}")

    def __iter__(self) -> Iterator[SecurityName]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class ByPeriod:
    _period_column_df: pl.DataFrame
    _security_column_df: pl.DataFrame = field(repr=False)
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    def as_df(self) -> pl.DataFrame:
        return self._period_column_df

    def latest(self) -> ArrayRowMapping:
        return self[-1]

    @overload
    def __getitem__(self, key: SupportsIndex) -> ArrayRowMapping: ...

    @overload
    def __getitem__(self, key: slice) -> PolarsPastView: ...

    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> ArrayRowMapping | PolarsPastView:
        if isinstance(key, SupportsIndex):
            vec = self._period_column_df.get_column(
                self._period_column_df.columns[key]
            ).to_numpy(allow_copy=False)
            return ArrayRowMapping(
                names=self._security_axis.names, data=vec, pos=self._security_axis.pos
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
                new_period_df, new_security_df, self._security_axis, new_period_axis
            )
        else:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}")


@dataclass(frozen=True)
class BySecurity:
    _security_column_df: pl.DataFrame
    _period_column_df: pl.DataFrame = field(repr=False)
    _security_axis: Axis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    def as_df(self) -> pl.DataFrame:
        return self._security_column_df

    @overload
    def __getitem__(self, key: SecurityName) -> NDArray[np.float64]: ...

    @overload
    def __getitem__(self, key: list) -> PolarsPastView: ...

    def __getitem__(
        self, key: SecurityName | list
    ) -> NDArray[np.float64] | PolarsPastView:
        if not isinstance(key, list):
            return self._security_column_df.get_column(key).to_numpy()

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
            by_period=ByPeriod(
                new_period_df, new_security_df, new_security_axis, self._period_axis
            ),
            by_security=BySecurity(
                new_security_df, new_period_df, new_security_axis, self._period_axis
            ),
            _period_axis=self._period_axis,
            _security_axis=new_security_axis,
        )


@dataclass(frozen=True)
class PolarsPastView:
    by_period: ByPeriod
    by_security: BySecurity
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
            dates = dates.cast(pl.Datetime("s"))

        period_names = dates.dt.to_string()
        non_date_cols = [x for x in df.columns if x != "date"]

        security_column_df = df.select(non_date_cols)
        period_column_df = security_column_df.transpose(column_names=period_names)

        security_axis = Axis.from_names(security_column_df.columns)
        period_axis = PeriodAxis.from_series(dates)

        return PolarsPastView(
            ByPeriod(period_column_df, security_column_df, security_axis, period_axis),
            BySecurity(
                security_column_df, period_column_df, security_axis, period_axis
            ),
            security_axis,
            period_axis,
        )

    def latest(self) -> ArrayRowMapping:
        return self.by_period.latest()

    def window(self, size: int, *, stagger: int = 0) -> PolarsPastView:
        return self.by_period[stagger : size + stagger]

    @property
    def size(self) -> int:
        return len(self.by_period.columns)
