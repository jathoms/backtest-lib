from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
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


@runtime_checkable
class UniverseMappingPlotAccessor(Protocol):
    def __init__(self, obj: VM): ...

    def __call__(
        self,
        *,
        kind: Literal["bar", "barh"] = "bar",
        **kwargs,
    ) -> Any: ...

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **kwargs,
    ) -> Any: ...

    def barh(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **kwargs,
    ) -> Any: ...

    def hist(
        self,
        bins: int | Iterable[float] = 20,
        select: int | slice | None = None,
        **kwargs,
    ) -> Any: ...


class TimeseriesPlotAccessor(Protocol):
    def __init__(self, obj: TS): ...

    def __call__(self, **kwargs) -> Any: ...

    def line(
        self,
        y_padding: float = 0.01,
        color: str = "steelblue",
        smoothing: int = 1,
        **kwargs,
    ) -> Any:
        """Plot the series as a line chart."""
        ...

    def kde(self, color="steelblue", **kwargs) -> Any: ...


class ByPeriodPlotAccessor(Protocol):
    def __init__(self, obj: BP): ...

    def __call__(self, **kwargs) -> Any:
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        securities: Sequence[SecurityName] | None = None,
        **kwargs,
    ) -> Any: ...

    def line(
        self,
        *,
        agg: Literal["mean", "median", "sum"],
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...

    def box(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...


class BySecurityPlotAccessor(Protocol):
    def __init__(self, obj: BS): ...

    def __call__(self, **kwargs) -> Any:
        return self.line(**kwargs)

    def line(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        agg: Literal["none", "mean", "median", "sum"] = "none",
        facet: bool = False,
        max_securities: int | None = None,
        **kwargs,
    ) -> Any: ...

    # - agg != "none": single aggregated line
    # - facet=True: one subplot per security (if small N)

    def heatmap(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...
