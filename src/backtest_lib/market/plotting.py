from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)

if TYPE_CHECKING:
    from backtest_lib.market import ByPeriod, BySecurity, PastView
    from backtest_lib.market.timeseries import Timeseries
    from backtest_lib.universe.vector_mapping import VectorMapping

BP = TypeVar("BP", bound="ByPeriod")
BS = TypeVar("BS", bound="BySecurity")
TS = TypeVar("TS", bound="Timeseries")
VM = TypeVar("VM", bound="VectorMapping")
PV = TypeVar("PV", bound="PastView")


class PastViewPlotAccessor(ABC):
    """Plot accessor for :class:`~backtest_lib.market.PastView` objects.

    Backends implement this interface to provide convenient plotting helpers for
    a full :class:`~backtest_lib.market.PastView`. The accessor is available on
    ``PastView.plot``.
    """

    @abstractmethod
    def __init__(self, obj): ...

    @abstractmethod
    def __call__(self, kind: Literal["bar", "line"] = "line", **kwargs): ...

    @abstractmethod
    def bar(
        self,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def line(
        self,
        agg: Literal["none", "mean", "sum"] = "none",
        y_padding: float = 0.01,
        smoothing: int = 1,
        **kwargs,
    ) -> Any: ...


class UniverseMappingPlotAccessor(ABC):
    """Plot accessor for
    :class:`~backtest_lib.universe.universe_mapping.UniverseMapping` data.

    Backends implement this interface to visualize a mapping of securities to
    scalar values. The accessor is available on ``UniverseMapping.plot``.
    """

    @abstractmethod
    def __init__(self, obj: VM): ...

    @abstractmethod
    def __call__(
        self,
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
    """Plot accessor for :class:`~backtest_lib.market.timeseries.Timeseries`.

    Implementations typically provide line and density plot helpers for a single
    security's time series data.
    """

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
    def bar(self, **kwargs) -> Any:
        """Plot the series as a bar chart."""
        ...

    @abstractmethod
    def hist(self, bins: int = 20, **kwargs) -> Any:
        """Histogram of the series values."""
        ...

    @abstractmethod
    def kde(self, color="steelblue", **kwargs) -> Any: ...


class ByPeriodPlotAccessor(ABC):
    """Plot accessor for :class:`~backtest_lib.market.ByPeriod` data.

    Exposes plotting helpers that treat the data as a sequence of per-period
    snapshots.
    """

    @abstractmethod
    def __init__(self, obj: BP): ...

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        return self.bar(**kwargs)

    @abstractmethod
    def bar(
        self,
        **kwargs,
    ) -> Any: ...


class BySecurityPlotAccessor(ABC):
    """Plot accessor for :class:`~backtest_lib.market.BySecurity` data.

    Exposes plotting helpers that treat the data as a collection of security
    time series.
    """

    @abstractmethod
    def __init__(self, obj: BS): ...

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        return self.line(**kwargs)

    @abstractmethod
    def line(
        self,
        agg: Literal["none", "mean", "sum"] = "none",
        y_padding: float = 0.01,
        smoothing: int = 1,
        **kwargs,
    ) -> Any: ...
