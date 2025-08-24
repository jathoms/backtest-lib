from __future__ import annotations
from backtest_lib.market.timeseries import Timeseries
from dataclasses import dataclass
from typing import TYPE_CHECKING
from backtest_lib.universe.vector_mapping import VectorMapping

if TYPE_CHECKING:
    from backtest_lib.market import PastView

from backtest_lib.market.timeseries import Comparable
from backtest_lib.universe.vector_mapping import SupportsFloat
from numbers import Real
from typing import TypeVar

R = TypeVar("R", bound=Real, contravariant=True)

SecurityName = str
Price = float
Volume = int


PeriodIndex = Comparable

Universe = tuple[SecurityName, ...]

type UniverseMapping[T: SupportsFloat] = VectorMapping[SecurityName, T]
type UniverseVolume = UniverseMapping[Volume]

type UniverseMask = VectorMapping[SecurityName, bool]
type UniversePriceView = PastView[UniverseMapping[Price], Timeseries, PeriodIndex]


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
