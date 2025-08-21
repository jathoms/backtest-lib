from __future__ import annotations
from numpy import datetime64
from collections.abc import Mapping
from backtest_lib.market.timeseries import Timeseries
from dataclasses import dataclass
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from backtest_lib.market import PastView

SecurityName = str
Price = float
Volume = float


T_co = TypeVar("T_co", covariant=True)


Universe = tuple[SecurityName]
type UniverseMapping[T_co] = Mapping[SecurityName, T_co]
type UniverseVolume = UniverseMapping[Volume]
type UniverseMask = UniverseMapping[bool]
type UniversePriceView = PastView[UniverseMapping[Price], Timeseries, datetime64]


@dataclass(frozen=True)
class PastUniversePrices:
    close: UniversePriceView
    open: UniversePriceView | None = None
    high: UniversePriceView | None = None
    low: UniversePriceView | None = None

    def truncated_to(self, n_periods: int) -> PastUniversePrices:
        return PastUniversePrices(
            close=self.close.by_period[:n_periods],
            open=self.open.by_period[:n_periods] if self.open else None,
            low=self.low.by_period[:n_periods] if self.low else None,
            high=self.high.by_period[:n_periods] if self.high else None,
        )
