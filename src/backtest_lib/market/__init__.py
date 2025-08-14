from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, TypeVar, runtime_checkable

from backtest_lib.universe import UniversePrices, UniverseVolume

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class PastView(Protocol[T_co]):
    """
    Time-fenced read-only series up to the current decision point.

    This protocol can abstract over different implementations (list-backed,
    NumPy array-backed, mmap, etc.) that present a "fenced" slice of
    historical snapshots ending at 'now', with no lookahead as to
    reduce the risk of lookahead bias while maintaining an ergonomic
    interface to access the market conditions.
    """

    def latest(self) -> T_co:
        """
        Return the latest snapshot in the view.

        This is always the last available element in the time-fenced
        series (index -1). For example, for a price series this would be
        the most recent bar's prices for the universe.
        """
        ...

    def window(self, size: int, *, stagger: int = 0) -> Sequence[T_co]:
        """
        Return a contiguous sequence of snapshots from the view.

        Args:
            size: Number of elements to return (must be > 0 and <= view size).
            stagger: How far back from the latest snapshot to start
                     (0 = end at latest; 1 = end one before latest, etc.).

        Example:
            view.window(10)       # last 10 snapshots ending at 'now'
            view.window(10, stagger=5)    # 10 snapshots ending 5 before 'now'
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


class MarketSnapshot:
    close_prices: UniversePrices
    volume: UniverseVolume | None


@dataclass(frozen=True)
class MarketView:
    close_prices: PastView[UniversePrices]
    volume: Optional[PastView[UniverseVolume]]
    # more stuff here

    def latest(self) -> MarketSnapshot:
        return MarketSnapshot(
            self.close_prices.latest(),
            self.volume.latest() if self.volume is not None else None,
        )
