from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import polars as pl

from backtest_lib.market import ByPeriod, BySecurity, PastView
from backtest_lib.market.plotting import (
    ByPeriodPlotAccessor,
    BySecurityPlotAccessor,
)
from backtest_lib.market.polars_impl._axis import PeriodAxis, SecurityAxis
from backtest_lib.market.polars_impl._helpers import (
    POLARS_TO_PYTHON,
    Array1DDTView,
    to_npdt64,
)
from backtest_lib.market.polars_impl._plotting import (
    PolarsByPeriodPlotAccessor,
    PolarsBySecurityPlotAccessor,
)
from backtest_lib.market.polars_impl._timeseries import PolarsTimeseries
from backtest_lib.market.polars_impl._universe_mapping import SeriesUniverseMapping
from backtest_lib.universe import SecurityName

if TYPE_CHECKING:
    import pandas as pd

    from backtest_lib.market import SecurityMappings

logger = logging.getLogger(__name__)

Scalar = TypeVar("Scalar", int, float)


@dataclass(frozen=True)
class PolarsByPeriod[ValueT: (float, int)](ByPeriod[ValueT, np.datetime64]):
    _period_column_df: pl.DataFrame
    _security_column_df: pl.DataFrame = field(repr=False)
    _security_axis: SecurityAxis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    _period_slice_start: int = field(repr=False, default=0)
    _period_slice_len: int | None = field(repr=False, default=None)  # None => to end

    _row_indexer: np.ndarray | None = field(repr=False, default=None)

    _col_names_cache: tuple[str, ...] = field(init=False, repr=False)

    @property
    def plot(self) -> ByPeriodPlotAccessor:
        return PolarsByPeriodPlotAccessor(self)

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

    @overload
    def as_df(
        self, *, show_securities: bool = ..., lazy: Literal[True] = ...
    ) -> pl.LazyFrame: ...

    @overload
    def as_df(
        self, *, show_securities: bool = ..., lazy: Literal[False] = ...
    ) -> pl.DataFrame: ...

    @overload
    def as_df(
        self, *, show_securities: bool = ..., lazy: bool = ...
    ) -> pl.DataFrame | pl.LazyFrame: ...

    def as_df(
        self, *, show_securities: bool = False, lazy: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        start = self._period_slice_start
        stop = self._period_slice_start + len(self)
        df = (
            self._period_column_df[:, start:stop].lazy()
            if lazy
            else self._period_column_df[:, start:stop]
        )

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
            return SeriesUniverseMapping(
                names=self._security_axis.names,
                _data=s,
                pos=self._security_axis.pos,
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
                _by_period=by_period_view,
                _by_security=by_security_view,
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
            _by_period=PolarsByPeriod(period_df, sec_df, self._security_axis, new_pax),
            _by_security=PolarsBySecurity(
                sec_df, period_df, self._security_axis, new_pax
            ),
            _period_axis=new_pax,
            _security_axis=self._security_axis,
        )

    def __iter__(self) -> Iterator[np.datetime64]:
        for period in self._period_axis.dt64:
            yield period

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[False] = False,
        backend: Literal["polars"],
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[False] = False,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[True],
        backend: Literal["polars"],
    ) -> pl.LazyFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: bool = ...,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    def to_dataframe(
        self,
        *,
        show_securities: bool = False,
        lazy: bool = False,
        backend: Literal["polars", "pandas"] = "polars",
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        if backend == "polars":
            return self.as_df(show_securities=show_securities, lazy=lazy)
        elif backend == "pandas":
            return self.as_df(show_securities=show_securities, lazy=False).to_pandas()
        raise ValueError(f"'{backend}' is not a valid DataFrame backend.")


@dataclass(frozen=True)
class PolarsBySecurity[ValueT: (float, int)](BySecurity[ValueT, np.datetime64]):
    _security_column_df: pl.DataFrame
    _period_column_df: pl.DataFrame = field(repr=False)
    _security_axis: SecurityAxis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    _period_slice_start: int = field(repr=False, default=0)
    _period_slice_len: int | None = field(repr=False, default=None)
    _sel_names: tuple[str, ...] | None = field(repr=False, default=None)

    def __len__(self) -> int:
        return len(self._security_axis)

    @overload
    def as_df(
        self, *, show_periods: bool = ..., lazy: Literal[True] = ...
    ) -> pl.LazyFrame: ...

    @overload
    def as_df(
        self, *, show_periods: bool = ..., lazy: Literal[False] = ...
    ) -> pl.DataFrame: ...

    @overload
    def as_df(
        self, *, show_periods: bool = ..., lazy: bool = ...
    ) -> pl.DataFrame | pl.LazyFrame: ...

    def as_df(
        self, *, show_periods: bool = True, lazy: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        df = self._security_column_df.lazy() if lazy else self._security_column_df
        if self._sel_names is not None:
            df = df.select(list(self._sel_names))
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

        new_security_axis = SecurityAxis.from_names(names)

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
            _by_period=by_period_view,
            _by_security=by_security_view,
            _period_axis=pax,
            _security_axis=new_security_axis,
        )

    def __iter__(self) -> Iterator[str]:
        for sec in self._security_axis.names:
            yield sec

    @property
    def plot(self) -> BySecurityPlotAccessor:
        return PolarsBySecurityPlotAccessor(self)

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: Literal[False] = ...,
        backend: Literal["polars"] = ...,
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: Literal[True],
        backend: Literal["polars"] = ...,
    ) -> pl.LazyFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: bool = ...,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    def to_dataframe(
        self,
        *,
        show_periods: bool = True,
        lazy: bool = False,
        backend: Literal["polars", "pandas"] = "polars",
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        if backend == "polars":
            return self.as_df(show_periods=show_periods, lazy=lazy)
        elif backend == "pandas":
            return self.as_df(show_periods=show_periods, lazy=False).to_pandas()
        raise ValueError(f"'{backend}' is not a valid DataFrame backend.")


@dataclass(frozen=True)
class PolarsPastView[ValueT: (float, int)](PastView[ValueT, np.datetime64]):
    _by_period: PolarsByPeriod = field(repr=False)
    _by_security: PolarsBySecurity = field()
    _security_axis: SecurityAxis = field(repr=False)
    _period_axis: PeriodAxis = field(repr=False)

    @property
    def by_period(self) -> PolarsByPeriod[ValueT]:
        return self._by_period

    @property
    def by_security(self) -> PolarsBySecurity[ValueT]:
        return self._by_security

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

        security_axis = SecurityAxis.from_names(security_column_df.columns)
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
            to_npdt64(start), inclusive=inclusive
        )
        return self._slice_period(left, right)

    def before(
        self, end: np.datetime64 | str, *, inclusive: bool = False
    ) -> PolarsPastView:
        left, right = self._period_axis.bounds_before(
            to_npdt64(end), inclusive=inclusive
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
            to_npdt64(start), to_npdt64(end), closed=closed
        )
        return self._slice_period(left, right)
