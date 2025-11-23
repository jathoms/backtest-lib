from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
)

from backtest_lib.market.timeseries import Timeseries
from backtest_lib.universe.vector_mapping import VectorMapping

if TYPE_CHECKING:
    from backtest_lib.market import ByPeriod, BySecurity

BP = TypeVar("BP", bound="ByPeriod")
BS = TypeVar("BS", bound="BySecurity")


class UniverseMappingPlotAccessor:
    def __init__(self, obj: VectorMapping):
        self._obj = obj

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def line(self, **style): ...


class ByPeriodPlotAccessor:
    def __init__(self, obj: BP):
        self._obj: ByPeriod = obj

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def line(self, **style): ...


class BySecurityPlotAccessor:
    def __init__(self, obj: BS):
        self._obj: BySecurity = obj

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def line(self, **style): ...


class TimeseriesPlotAccessor:
    def __init__(self, obj: Timeseries):
        self._obj = obj

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **style,
    ): ...

    def line(self, **style): ...
