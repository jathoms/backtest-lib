from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Iterator, Self, Sequence, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

POLARS_TO_PYTHON: dict[pl.DataType | type[Any], type[Any]] = {
    pl.Boolean: bool,
    pl.Int8: int,
    pl.Int16: int,
    pl.Int32: int,
    pl.Int64: int,
    pl.UInt8: int,
    pl.UInt16: int,
    pl.UInt32: int,
    pl.UInt64: int,
    pl.Float32: float,
    pl.Float64: float,
    pl.String: str,
    pl.Categorical: str,  # categorical values are stored as strings
    pl.Enum: str,
    pl.Date: dt.date,
    pl.Datetime: dt.datetime,
    pl.Time: dt.time,
    pl.Duration: dt.timedelta,
    pl.Decimal: Decimal,
    pl.Binary: bytes,
    pl.Object: object,
    pl.Null: type(None),
}


class Array1DDTView(Sequence[np.datetime64]):
    """
    Zero-copy 1-D Sequence view over an NDArray[np.datetime64].
    Ensures 1-D at construction; slicing returns another view.

    This serves mainly as a wrapper around NDArray that implements
    Sequence.
    """

    def __init__(self, a: NDArray[np.datetime64]):
        self._a = np.asarray(a.reshape(-1), copy=False) if a.ndim != 1 else a

    @property
    def array(self) -> NDArray[np.datetime64]:
        return self._a

    def __len__(self) -> int:
        return self._a.shape[0]

    @overload
    def __getitem__(self, index: int) -> np.datetime64: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> np.datetime64 | Self:
        if isinstance(index, slice):
            return Array1DDTView(self._a[index])
        if isinstance(index, (int, np.integer)):
            return self._a[index]
        raise TypeError(f"Invalid index type: {type(index)!r}")

    def __iter__(self) -> Iterator[np.datetime64]:
        for i in range(self._a.shape[0]):
            yield self._a[i]

    def __repr__(self) -> str:
        return f"Array1DDTView({self._a!r})"

    def __array__(self) -> NDArray[np.datetime64]:
        return self._a.astype("datetime64[us]")


def to_npdt64(x: np.datetime64 | str) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[us]")
    if isinstance(x, (str, dt.datetime, dt.date)):
        return np.datetime64(x, "us")
    raise TypeError(f"Unsupported type {type(x)}")
