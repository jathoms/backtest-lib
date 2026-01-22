from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from backtest_lib.market import Closed


@dataclass(frozen=True)
class SecurityAxis:
    names: tuple[str, ...]
    pos: dict[str, int]  # name -> index (0..N-1)

    @staticmethod
    def from_names(names: Sequence[str]) -> SecurityAxis:
        names_t = tuple(names)
        return SecurityAxis(names_t, {s: i for i, s in enumerate(names_t)})

    def __len__(self):
        return len(self.names)


@dataclass(frozen=True)
class PeriodAxis:
    dt64: NDArray[np.datetime64]
    labels: tuple[str, ...]
    pos: dict[str, int]  # label -> index

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def from_series(date_s: pl.Series, fmt: str = "%Y-%m-%d") -> PeriodAxis:
        if date_s.dtype not in (pl.Date, pl.Datetime):
            date_s = date_s.cast(pl.Datetime("us"))
        labels = tuple(date_s.dt.strftime(fmt).to_list())
        dt64 = date_s.to_numpy().astype("datetime64[us]", copy=False)
        return PeriodAxis(dt64, labels, {lbl: i for i, lbl in enumerate(labels)})

    def take(self, idxs: Sequence[int] | NDArray[np.integer]) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a sequence of
        integer indices contained in the period axis.
        """
        new_labels = tuple(self.labels[i] for i in idxs)
        return PeriodAxis(
            dt64=self.dt64[idxs],
            labels=new_labels,
            pos={lbl: i for i, lbl in enumerate(new_labels)},
        )

    def slice(self, key: slice) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a slice
        of the current PeriodAxis.
        """
        start, stop, step = key.indices(len(self))
        if step == 1:
            return self.slice_contiguous(start, stop)
        period_idxs = np.arange(start, stop, step, dtype=np.int64)
        return self.take(period_idxs)

    def slice_contiguous(self, start: int, stop: int) -> PeriodAxis:
        """
        Creates a new PeriodAxis from a slice
        of the current PeriodAxis.
        """
        new_dt64 = self.dt64[start:stop]
        new_labels = self.labels[start:stop]
        new_pos = {lbl: i for i, lbl in enumerate(new_labels)}
        return PeriodAxis(new_dt64, new_labels, new_pos)

    def bounds_after(
        self, start: np.datetime64, *, inclusive: bool = True
    ) -> tuple[int, int]:
        left = int(
            np.searchsorted(self.dt64, start, side="left" if inclusive else "right")
        )
        return left, len(self.dt64)

    def bounds_before(
        self, end: np.datetime64, *, inclusive: bool = False
    ) -> tuple[int, int]:
        right = int(
            np.searchsorted(self.dt64, end, side="right" if inclusive else "left")
        )
        return 0, right

    def bounds_between(
        self,
        start: np.datetime64,
        end: np.datetime64,
        *,
        closed: str | Closed = Closed.BOTH,
    ) -> tuple[int, int]:
        inc_start = closed in (Closed.BOTH, Closed.LEFT)
        inc_end = closed in (Closed.BOTH, Closed.RIGHT)
        left, _ = self.bounds_after(start, inclusive=inc_start)
        _, right = self.bounds_before(end, inclusive=inc_end)
        return left, right
