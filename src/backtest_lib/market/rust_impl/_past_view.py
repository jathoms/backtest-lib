from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, TypeVar, cast, overload

import numpy as np
import polars as pl

from backtest_lib.market import (
    ByPeriod,
    BySecurity,
    Closed,
    PastView,
    SecuritySelection,
)
from backtest_lib.market.plotting import (
    ByPeriodPlotAccessor,
    BySecurityPlotAccessor,
    PastViewPlotAccessor,
)
from backtest_lib.market.polars_impl._past_view import (
    PolarsByPeriod,
    PolarsBySecurity,
    PolarsPastView,
)
from backtest_lib.market.polars_impl._timeseries import PolarsTimeseries
from backtest_lib.market.rust_impl._universe_mapping import NativeUniverseMapping

if TYPE_CHECKING:
    import pandas as pd

    from backtest_lib.market import SecurityMappings

Scalar = TypeVar("Scalar", int, float)


@overload
def _wrap_past_view(inner: PolarsPastView[int]) -> NativePastView[int]: ...


@overload
def _wrap_past_view(inner: PolarsPastView[float]) -> NativePastView[float]: ...


def _wrap_past_view(
    inner: PolarsPastView[int] | PolarsPastView[float],
) -> NativePastView[int] | NativePastView[float]:
    return cast(
        NativePastView[int] | NativePastView[float],
        NativePastView(cast(Any, inner)),
    )


@dataclass(frozen=True)
class NativeByPeriod[ValueT: (float, int)](ByPeriod[ValueT, np.datetime64]):
    inner: PolarsByPeriod[ValueT]

    def _get_universe_mapping(
        self, key: SupportsIndex
    ) -> NativeUniverseMapping[ValueT]:
        return NativeUniverseMapping.from_polars_mapping(self.inner[key])

    def _get_slice(self, key: slice) -> NativePastView[ValueT]:
        return _wrap_past_view(self.inner[key])

    @property
    def plot(self) -> ByPeriodPlotAccessor:
        return self.inner.plot

    def __len__(self) -> int:
        return len(self.inner)

    @overload
    def as_df(
        self, *, show_securities: bool = ..., lazy: Literal[True]
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
        return self.inner.as_df(show_securities=show_securities, lazy=lazy)

    @overload
    def __getitem__(self, key: SupportsIndex) -> NativeUniverseMapping[ValueT]: ...

    @overload
    def __getitem__(self, key: slice) -> NativePastView[ValueT]: ...

    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> NativeUniverseMapping[ValueT] | NativePastView[ValueT]:
        if isinstance(key, slice):
            return self._get_slice(key)
        return self._get_universe_mapping(key)

    def __iter__(self) -> Iterator[np.datetime64]:
        return iter(self.inner)

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
        return self.inner.to_dataframe(
            show_securities=show_securities,
            lazy=lazy,
            backend=backend,
        )


@dataclass(frozen=True)
class NativeBySecurity[ValueT: (float, int)](BySecurity[ValueT, np.datetime64]):
    inner: PolarsBySecurity[ValueT]

    def _get_timeseries(self, key: str) -> PolarsTimeseries[ValueT]:
        return self.inner[key]

    def _get_slice(self, key: SecuritySelection) -> NativePastView[ValueT]:
        return _wrap_past_view(self.inner[key])

    def __len__(self) -> int:
        return len(self.inner)

    @overload
    def as_df(
        self, *, show_periods: bool = ..., lazy: Literal[True]
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
        return self.inner.as_df(show_periods=show_periods, lazy=lazy)

    @overload
    def __getitem__(self, key: str) -> PolarsTimeseries[ValueT]: ...

    @overload
    def __getitem__(self, key: SecuritySelection) -> NativePastView[ValueT]: ...

    def __getitem__(
        self, key: str | SecuritySelection
    ) -> PolarsTimeseries[ValueT] | NativePastView[ValueT]:
        if isinstance(key, str):
            return self._get_timeseries(key)
        return self._get_slice(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner)

    @property
    def plot(self) -> BySecurityPlotAccessor:
        return self.inner.plot

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
        return self.inner.to_dataframe(
            show_periods=show_periods,
            lazy=lazy,
            backend=backend,
        )


@dataclass(frozen=True)
class NativePastView[ValueT: (float, int)](PastView[ValueT, np.datetime64]):
    inner: PolarsPastView[ValueT]

    @staticmethod
    def _from_inner(inner: PolarsPastView[ValueT]) -> NativePastView[ValueT]:
        return NativePastView(inner)

    @property
    def by_period(self) -> NativeByPeriod[ValueT]:
        return NativeByPeriod(self.inner.by_period)

    @property
    def by_security(self) -> NativeBySecurity[ValueT]:
        return NativeBySecurity(self.inner.by_security)

    @property
    def periods(self) -> tuple[np.datetime64, ...]:
        return self.inner.periods

    @property
    def securities(self) -> tuple[str, ...]:
        return self.inner.securities

    @staticmethod
    def from_security_mappings(
        ms: SecurityMappings[int] | SecurityMappings[float],
        periods: Sequence[np.datetime64],
    ) -> NativePastView[int] | NativePastView[float]:
        return _wrap_past_view(PolarsPastView.from_security_mappings(ms, periods))

    @staticmethod
    def from_dataframe(df: pl.DataFrame | pd.DataFrame) -> NativePastView:
        return NativePastView._from_inner(PolarsPastView.from_dataframe(df))

    def after(
        self, start: np.datetime64 | str, *, inclusive: bool = True
    ) -> NativePastView[ValueT]:
        return NativePastView._from_inner(self.inner.after(start, inclusive=inclusive))

    def before(
        self, end: np.datetime64 | str, *, inclusive: bool = False
    ) -> NativePastView[ValueT]:
        return NativePastView._from_inner(self.inner.before(end, inclusive=inclusive))

    def between(
        self,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        closed: Closed | str = Closed.LEFT,
    ) -> NativePastView[ValueT]:
        return NativePastView._from_inner(self.inner.between(start, end, closed=closed))

    @property
    def plot(self) -> PastViewPlotAccessor:
        return self.inner.plot
