from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from backtest_lib.market import ByPeriod, BySecurity
    from backtest_lib.market.timeseries import Index, Timeseries
    from backtest_lib.universe import SecurityName
    from backtest_lib.universe.vector_mapping import VectorMapping

BP = TypeVar("BP", bound="ByPeriod")
BS = TypeVar("BS", bound="BySecurity")
TS = TypeVar("TS", bound="Timeseries")
VM = TypeVar("VM", bound="VectorMapping")


class UniverseMappingPlotAccessor:
    def __init__(self, obj: VM):
        self._obj: VectorMapping = obj

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


class TimeseriesPlotAccessor:
    def __init__(self, obj: TS):
        self._obj: Timeseries = obj

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


class ByPeriodPlotAccessor:
    def __init__(self, obj: BP):
        self._obj: ByPeriod = obj

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


class BySecurityPlotAccessor:
    def __init__(self, obj: BS):
        self._obj: BySecurity = obj

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
