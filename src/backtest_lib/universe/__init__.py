from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from backtest_lib.market import PastView
    from backtest_lib.market.timeseries import Comparable
    from backtest_lib.universe.vector_mapping import VectorMapping


SecurityName = str
Price = float
Volume = int


Universe = tuple[SecurityName, ...]


type UniversePriceView[TPeriod: Comparable] = PastView[Price, TPeriod]


@dataclass(frozen=True)
class PastUniversePrices[Index: Comparable]:
    close: UniversePriceView[Index]
    open: UniversePriceView[Index] | None = None
    high: UniversePriceView[Index] | None = None
    low: UniversePriceView[Index] | None = None

    def truncated_to(self, n_periods: int) -> Self:
        return PastUniversePrices(
            close=self.close.by_period[:n_periods],
            open=self.open.by_period[:n_periods] if self.open else None,
            low=self.low.by_period[:n_periods] if self.low else None,
            high=self.high.by_period[:n_periods] if self.high else None,
        )

    def filter_securities(self, securities: Sequence[SecurityName]) -> Self:
        return PastUniversePrices(
            close=self.close.by_security[securities],
            open=self.open.by_security[securities] if self.open else None,
            low=self.low.by_security[securities] if self.low else None,
            high=self.high.by_security[securities] if self.high else None,
        )
