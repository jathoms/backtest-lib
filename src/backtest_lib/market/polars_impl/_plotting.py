from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Sequence,
)

from backtest_lib.market.polars_impl._past_view import PolarsByPeriod, PolarsBySecurity
from backtest_lib.market.polars_impl._timeseries import PolarsTimeseries
from backtest_lib.market.polars_impl._universe_mapping import SeriesUniverseMapping

if TYPE_CHECKING:
    from backtest_lib.market.timeseries import Index
    from backtest_lib.universe import SecurityName


class SeriesUniverseMappingPlotAccessor:
    def __init__(self, obj: SeriesUniverseMapping):
        self._obj = obj

    def __call__(
        self,
        *,
        kind: Literal["bar", "barh"] = "bar",
        **kwargs,
    ):
        if kind == "bar":
            return self.bar(**kwargs)
        elif kind == "barh":
            return self.barh(**kwargs)
        raise ValueError(kind)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def barh(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def hist(
        self,
        bins: int | Iterable[float] = 20,
        select: int | slice | None = None,
        **style,
    ): ...


class PolarsTimeseriesPlotAccessor:
    def __init__(self, obj: PolarsTimeseries):
        self._obj = obj

    def __call__(self, **style):
        return self.line(**style)

    def line(self, **style):
        """Plot the series as a line chart."""
        ...

    def bar(self, **style):
        """Plot the series as a bar chart (e.g. returns)."""
        ...

    def hist(self, bins: int | Sequence[float] = 20, **style):
        """Histogram of the values."""
        ...

    def accum_line(self, **style):
        """Cumulative version (if that's a common semantic op)."""
        ...


class PolarsByPeriodPlotAccessor:
    def __init__(self, obj: PolarsByPeriod):
        self._obj = obj

    def __call__(self, **kwargs):
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        securities: Sequence[SecurityName] | None = None,
        **style,
    ): ...

    def line(
        self,
        *,
        agg: Literal["mean", "median", "sum"],
        periods: slice | Sequence[Index] | None = None,
        **style,
    ): ...

    def box(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        **style,
    ): ...


class PolarsBySecurityPlotAccessor:
    def __init__(self, obj: PolarsBySecurity):
        self._obj = obj

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def line(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        agg: Literal["none", "mean", "median", "sum"] = "none",
        facet: bool = False,
        max_securities: int | None = None,
        **style,
    ): ...

    # - agg != "none": single aggregated line
    # - facet=True: one subplot per security (if small N)

    def heatmap(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        periods: slice | Sequence[Index] | None = None,
        **style,
    ): ...
