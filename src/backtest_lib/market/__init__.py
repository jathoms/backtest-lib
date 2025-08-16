from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable, overload, SupportsIndex

from backtest_lib.universe import (
    UniverseVolume,
    UniverseMask,
    UniversePrices,
)

T = TypeVar("T", covariant=True)


@runtime_checkable
class PastView(Protocol[T]):
    """
    Time-fenced read-only series up to the current decision point.

    This protocol can abstract over different implementations (list-backed,
    NumPy array-backed, mmap, etc.) that present a "fenced" slice of
    historical snapshots ending at "now", with no lookahead as to
    reduce the risk of lookahead bias while maintaining an ergonomic
    interface to access the market conditions.
    """

    def latest(self) -> T:
        """
        Return the latest snapshot in the view.

        This is always the last available element in the time-fenced
        series (index -1). For example, for a price series this would be
        the most recent bar's prices for the universe.
        """
        ...

    def window(self, size: int, *, stagger: int = 0) -> PastView[T]:
        """
        Return a contiguous sequence of snapshots from the view.

        Args:
            size: Number of elements to return (must be > 0 and <= view size).
            stagger: How far back from the latest snapshot to start
                     (0 = end at latest; 1 = end one before latest, etc.).

        Example:
            view.window(10)               # last 10 snapshots ending at "now"
            view.window(10, stagger=5)    # 10 snapshots ending 5 before "now"
        """
        ...

    @property
    def size(self) -> int:
        """
        Number of snapshots in the view.

        This counts how many time steps are available in the current
        fence. Equal to the maximum `size` you can request from `window()`.
        """
        ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> PastView[T]: ...


@dataclass(frozen=True)
class MarketSnapshot:
    prices: UniversePrices
    volume: UniverseVolume | None
    tradable: UniverseMask


@dataclass(frozen=True)
class MarketView:
    prices: PastView[UniversePrices]
    volume: PastView[UniverseVolume] | None
    tradable: PastView[UniverseMask]
    # more stuff here

    def latest(self) -> MarketSnapshot:
        return MarketSnapshot(
            self.prices.latest(),
            self.volume.latest() if self.volume is not None else None,
            self.tradable.latest(),
        )

    def window(self, size: int, *, stagger: int = 0) -> MarketView:
        return MarketView(
            prices=self.prices.window(size, stagger=stagger),
            volume=self.volume.window(size, stagger=stagger)
            if self.volume is not None
            else None,
            tradable=self.tradable.window(size, stagger=stagger),
        )

    def size(self) -> int:
        return min(v.size for v in self.__dict__.values() if isinstance(v, PastView))

    @overload
    def __getitem__(self, key: SupportsIndex) -> MarketSnapshot: ...

    @overload
    def __getitem__(self, key: slice) -> MarketView: ...

    def __getitem__(self, key: SupportsIndex | slice) -> MarketSnapshot | MarketView:
        if isinstance(key, SupportsIndex):
            return MarketSnapshot(
                self.prices[key], self.volume[key], self.tradable[key]
            )
        elif isinstance(key, slice):
            return MarketView(self.prices[key], self.volume[key], self.tradable[key])
        else:
            raise ValueError(
                f"Unsupported index '{key}' ({key.__name__}) with type {type(key)}"
            )
