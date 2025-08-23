from __future__ import annotations
from collections.abc import Sequence

from dataclasses import dataclass
from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
    overload,
    SupportsIndex,
    Any,
    Self,
)

from backtest_lib.universe import (
    UniverseVolume,
    UniverseMask,
    UniverseMapping,
    PastUniversePrices,
    SecurityName,
    PeriodIndex,
)

from backtest_lib.market.timeseries import Timeseries
from backtest_lib.market.timeseries import Comparable


Index = TypeVar("Index", bound=Comparable)

S = TypeVar(
    "S", bound=UniverseMapping, covariant=True
)  # mapping of securities to prices
P = TypeVar(
    "P", bound=Timeseries[Any, Comparable], covariant=True
)  # mapping of periods to some data (prices, volume, is_tradable)


@runtime_checkable
class PastView(Protocol[S, P, Index]):
    """
    Time-fenced read-only series up to the current decision point.

    This protocol can abstract over different implementations (list-backed,
    NumPy array-backed, mmap, etc.) that present a "fenced" slice of
    historical snapshots ending at "now", with no lookahead as to
    reduce the risk of lookahead bias while maintaining an ergonomic
    interface to access the market conditions.
    """

    @property
    def by_period(self) -> ByPeriod[S, P, Index]: ...

    @property
    def by_security(self) -> BySecurity[S, P, Index]: ...

    def between(
        self,
        start: Index | str,
        end: Index | str,
    ) -> Self: ...  # will not clone data, must be contiguous, performs a binary search

    def after(
        self,
        start: Index | str,
        *,
        inclusive: bool = True,  # common expectation: include the start tick
    ) -> Self: ...

    def before(
        self,
        end: Index | str,
        *,
        inclusive: bool = False,  # common expectation: half-open [.., end)
    ) -> Self: ...


@runtime_checkable
class ByPeriod(Protocol[S, P, Index]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> S: ...

    @overload
    def __getitem__(self, key: slice) -> PastView[S, P, Index]: ...


@runtime_checkable
class BySecurity(Protocol[S, P, Index]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SecurityName) -> P: ...

    @overload
    def __getitem__(self, key: list[SecurityName]) -> PastView[S, P, Index]: ...


@dataclass(frozen=True)
class MarketView:
    prices: PastUniversePrices
    periods: Sequence[PeriodIndex]
    tradable: PastView[UniverseMask, Timeseries, PeriodIndex] | None = None
    volume: PastView[UniverseVolume, Timeseries, PeriodIndex] | None = None

    def truncated_to(self, n_periods: int) -> MarketView:
        return MarketView(
            prices=self.prices.truncated_to(n_periods),
            volume=self.volume.by_period[:n_periods] if self.volume else None,
            tradable=self.tradable.by_period[:n_periods] if self.tradable else None,
            periods=self.periods[:n_periods],
        )
