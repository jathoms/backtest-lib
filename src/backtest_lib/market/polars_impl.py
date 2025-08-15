from __future__ import annotations
from dataclasses import dataclass, field
import polars as pl
import numpy as np
from typing import Sequence
from collections.abc import Iterator, Mapping
from numpy.typing import NDArray
from backtest_lib.universe import SecurityName, Universe


@dataclass(frozen=True)
class Axis:
    names: tuple[SecurityName, ...]
    pos: dict[SecurityName, int]  # name -> index (0..N-1)

    @staticmethod
    def from_names(names: Sequence[SecurityName]) -> Axis:
        names_t = tuple(names)
        return Axis(names_t, {s: i for i, s in enumerate(names_t)})


@dataclass(frozen=True)
class ArrayRowMapping(Mapping[SecurityName, float]):
    names: Universe
    data: NDArray[np.float64]  # (N,)
    pos: dict[SecurityName, int] = field(repr=False)

    def __post_init__(self):
        # dtype/shape checks
        if 1 not in self.data.shape:
            raise ValueError("data must be 1-D")

        self.data.reshape(-1)

        if self.data.dtype != np.float64:
            # normalize once so downstream math is predictable
            object.__setattr__(self, "data", self.data.astype(np.float64, copy=False))
        n = len(self.names)
        if len(self.data) != n or len(self.pos) != n:
            raise ValueError(
                f"Row mapping misaligned: names={n}, data={len(self.data)}, pos={len(self.pos)}"
            )
        # ensure pos matches the declared order
        if any(self.pos[name] != i for i, name in enumerate(self.names)):
            raise ValueError("pos mapping does not match names order")
        # expose as read-only
        self.data.setflags(write=False)

    def __getitem__(self, k: SecurityName) -> float:
        return float(self.data[self.pos[k]])

    def __iter__(self) -> Iterator[SecurityName]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def get_many(self, keys: list[SecurityName]) -> NDArray[np.float64]:
        idx = np.fromiter((self.pos[k] for k in keys), dtype=np.int64, count=len(keys))
        return self.data[idx]


@dataclass(frozen=True)
class PolarsPastView:
    _inner_df: pl.DataFrame
    _axis: Axis
    _dates: NDArray[np.datetime64]

    @staticmethod
    def from_data_frame(df: pl.DataFrame) -> PolarsPastView:
        try:
            dates = df.get_column("date", default=None)
            dates = dates.to_numpy()
        except Exception as e:
            raise ValueError(
                "Input dataframe must have column for 'date' and a column for each security"
            ) from e

        non_date_cols = [x for x in df.columns if x != "date"]

        axis = Axis.from_names(non_date_cols)
        df = df.select(non_date_cols).transpose()
        df = df.rename(
            {orig: orig.replace("column_", "period_") for orig in df.columns}
        )

        return PolarsPastView(df, axis, dates)

    def latest(self) -> ArrayRowMapping:
        vec = self._inner_df.select(self._inner_df.columns[0]).to_numpy()
        return ArrayRowMapping(names=self._axis.names, data=vec, pos=self._axis.pos)

    def window(self, size: int, *, stagger: int = 0) -> PolarsPastView:
        new_inner_df = self._inner_df.select(
            self._inner_df.columns[stagger : size + stagger]
        )
        return PolarsPastView(
            new_inner_df,
            self._axis,
            self._dates[stagger : size + stagger],
        )

    @property
    def size(self) -> int:
        return len(self._inner_df.columns)

    def truncate(self, period: int) -> PolarsPastView:
        return self.window(self.size - period, stagger=period)
