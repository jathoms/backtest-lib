from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from backtest_lib.market import PastView
    from backtest_lib.market.timeseries import Comparable


Universe = tuple[str, ...]


@dataclass(frozen=True)
class PastUniversePrices[Index: Comparable]:
    """PLACEHOLDER"""

    close: PastView[float, Index]
    open: PastView[float, Index] | None = None
    high: PastView[float, Index] | None = None
    low: PastView[float, Index] | None = None

    def truncated_to(self, n_periods: int) -> Self:
        return PastUniversePrices(
            close=self.close.by_period[:n_periods],
            open=self.open.by_period[:n_periods] if self.open else None,
            low=self.low.by_period[:n_periods] if self.low else None,
            high=self.high.by_period[:n_periods] if self.high else None,
        )

    def filter_securities(self, securities: Sequence[str]) -> Self:
        return PastUniversePrices(
            close=self.close.by_security[securities],
            open=self.open.by_security[securities] if self.open else None,
            low=self.low.by_security[securities] if self.low else None,
            high=self.high.by_security[securities] if self.high else None,
        )
