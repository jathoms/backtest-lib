from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
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


class UniverseMappingPlotAccessor(ABC):
    @abstractmethod
    def __init__(self, obj: VM): ...

    @abstractmethod
    def __call__(
        self,
        *,
        kind: Literal["bar", "barh"] = "bar",
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def bar(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        color: str = "steelblue",
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def barh(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        color: str = "steelblue",
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def stacked_bar(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        bar_label: str = "",
        **kwargs,
    ) -> Any: ...


class TimeseriesPlotAccessor(ABC):
    @abstractmethod
    def __init__(self, obj: TS): ...

    @abstractmethod
    def __call__(self, **kwargs) -> Any: ...

    @abstractmethod
    def line(
        self,
        y_padding: float = 0.01,
        color: str = "steelblue",
        smoothing: int = 1,
        **kwargs,
    ) -> Any:
        """Plot the series as a line chart."""
        ...

    @abstractmethod
    def kde(self, color="steelblue", **kwargs) -> Any: ...


class ByPeriodPlotAccessor(ABC):
    @abstractmethod
    def __init__(self, obj: BP): ...

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        return self.heatmap(**kwargs)

    @abstractmethod
    def heatmap(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        securities: Sequence[SecurityName] | None = None,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def line(
        self,
        *,
        agg: Literal["mean", "median", "sum"],
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def box(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...


class BySecurityPlotAccessor(ABC):
    @abstractmethod
    def __init__(self, obj: BS): ...

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        return self.line(**kwargs)

    @abstractmethod
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

    @abstractmethod
    def heatmap(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ) -> Any: ...
