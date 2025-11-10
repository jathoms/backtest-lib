from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backtest_lib.market.timeseries import Timeseries
from backtest_lib.universe.vector_mapping import VectorMapping

if TYPE_CHECKING:
    from backtest_lib.market import PastView

from backtest_lib.market.timeseries import Comparable

SecurityName = str
Price = float
Volume = int


PeriodIndex = Comparable

Universe = tuple[SecurityName, ...]

type UniverseMapping[T: (int, float)] = VectorMapping[SecurityName, T]
type UniverseVolume = UniverseMapping[Volume]
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

    def filter_securities(
        self, securities: Sequence[SecurityName]
    ) -> PastUniversePrices:
        return PastUniversePrices(
            close=self.close.by_security[securities],
            open=self.open.by_security[securities] if self.open else None,
            low=self.low.by_security[securities] if self.low else None,
            high=self.high.by_security[securities] if self.high else None,
        )
