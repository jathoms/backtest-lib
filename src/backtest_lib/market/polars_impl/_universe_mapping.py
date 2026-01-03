from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    Self,
    SupportsFloat,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import polars as pl

from backtest_lib.market.plotting import (
    UniverseMappingPlotAccessor,
)
from backtest_lib.market.polars_impl._helpers import POLARS_TO_PYTHON
from backtest_lib.market.polars_impl._plotting import SeriesUniverseMappingPlotAccessor
from backtest_lib.universe import Universe
from backtest_lib.universe.universe_mapping import UniverseMapping
from backtest_lib.universe.vector_mapping import VectorMapping
from backtest_lib.universe.vector_ops import VectorOps

_RHS_HANDOFF = object()
logger = logging.getLogger(__name__)

Other_scalar = TypeVar("Other_scalar", float, int)
ScalarU = float | int


@dataclass(frozen=True, init=False)
class SeriesUniverseMapping[T: (float, int)](UniverseMapping[T]):
    names: Universe
    _data: pl.Series
    pos: dict[str, int] = field(repr=False)
    _scalar_type: type[T]

    @staticmethod
    def from_names_and_data(
        names: Universe,
        data: pl.Series,
        dtype: type[int] | type[float] | None = None,
    ) -> Self:
        return SeriesUniverseMapping(
            names=names,
            _data=data,
            pos={name: i for i, name in enumerate(names)},
            _scalar_type=dtype,
        )

    def __init__(
        self,
        names: Universe,
        _data: pl.Series,
        pos: dict[str, int],
        _scalar_type: type[int] | type[float] | None = None,
    ):
        n = len(names)
        if len(_data) != n or len(pos) != n:
            raise ValueError(
                f"Row mapping misaligned: names={n}, _data={len(_data)}, pos={len(pos)}"
            )
        if _scalar_type is None:
            _scalar_type = POLARS_TO_PYTHON[_data.dtype]
        if _scalar_type is float and _data.dtype != pl.Float64:
            _data = _data.cast(pl.Float64)
        elif _scalar_type is int and _data.dtype != pl.Int64:
            _data = _data.cast(pl.Int64)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "_data", _data)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "_scalar_type", _scalar_type)

    def as_series(self) -> pl.Series:
        return self._data

    @overload
    def __getitem__(self, key: str) -> T: ...

    @overload
    def __getitem__(self, key: Iterable[str]) -> pl.Series: ...

    def __getitem__(self, key: str | Iterable[str]) -> T | pl.Series:
        if isinstance(key, str):
            return self._scalar_type(self._data.item(self.pos[key]))
        elif isinstance(key, list):
            idx = np.fromiter(
                (self.pos[k] for k in key), dtype=np.int64, count=len(key)
            )
            return self._data.gather(idx)
        else:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def _rhs(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> tuple[pl.Series | float, type[float] | type[int]]:
        if isinstance(other, SupportsFloat):
            if not isinstance(other, int) and self._scalar_type is int:
                return float(other), float
            return float(other), self._scalar_type
        elif isinstance(other, SeriesUniverseMapping):
            data = other._data
            if other.names != self.names:
                if not all(name in self.names for name in other.names):
                    logger.debug(f"lhs size: {len(self)}, rhs size: {len(other)}")
                    logger.debug(
                        f"{
                            len(
                                [name for name in self.names if name not in other.names]
                            )
                        }"
                        " items found in lhs not in rhs"
                    )
                    return cast(float, _RHS_HANDOFF), float
                data = _mapping_to_series(self, other)
            if other._scalar_type is not int:
                return data, float
            return data, self._scalar_type
        elif isinstance(other, Mapping):
            aligned_series = _mapping_to_series(self, other)
            if (
                any(not isinstance(x, int) for x in other.values())
                and self._scalar_type is int
            ):
                return aligned_series, float
            return aligned_series, self._scalar_type

        raise TypeError(
            "Unsupported operand: only scalars, mappings or SeriesUniverseMapping "
            "with compatible axes are allowed."
        )

    def __add__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            return self.__radd__(other)
        return SeriesUniverseMapping(self.names, self._data + rhs, self.pos, new_type)

    def __radd__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs + self._data, self.pos, new_type)

    def __sub__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rsub__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping(self.names, self._data - rhs, self.pos, new_type)

    def __rsub__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs - self._data, self.pos, new_type)

    def __mul__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        rhs, new_type = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rmul__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping[new_type](
            self.names, self._data * rhs, self.pos, new_type
        )

    def __rmul__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping:
        lhs, new_type = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping(self.names, lhs * self._data, self.pos, new_type)

    def __truediv__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping[float]:
        rhs, _ = self._rhs(other)
        if rhs is _RHS_HANDOFF:
            if isinstance(other, SeriesUniverseMapping):
                return other.__rtruediv__(self)
            else:
                return NotImplemented
        return SeriesUniverseMapping[float](
            self.names, self._data / rhs, self.pos, float
        )

    def __rtruediv__(
        self, other: VectorOps[Other_scalar] | ScalarU | Mapping
    ) -> SeriesUniverseMapping[float]:
        lhs, _ = self._rhs(other)
        if lhs is _RHS_HANDOFF:
            return NotImplemented
        return SeriesUniverseMapping[float](
            self.names, lhs / self._data, self.pos, float
        )

    def sum(self) -> T:
        return self._scalar_type(self._data.sum())

    def mean(self) -> T:
        # assert self._data.dtype == pl.Float64
        dt = self._data.dtype
        if not (dt.is_numeric()):
            raise TypeError(f"mean() only supported on numeric dtypes, got {dt!r}")
        m = self._data.mean()
        if m is None:
            raise ValueError("Mean of empty series")
        return self._scalar_type(m)

    def abs(self) -> Self:
        return SeriesUniverseMapping(
            self.names, self._data.abs(), self.pos, self._scalar_type
        )

    def floor(self) -> SeriesUniverseMapping[int]:
        return SeriesUniverseMapping[int](
            names=self.names, _data=self._data.floor(), pos=self.pos, _scalar_type=int
        )

    def truncate(self) -> SeriesUniverseMapping[int]:
        return SeriesUniverseMapping[int](
            names=self.names,
            _data=self._data.cast(pl.Int64),
            pos=self.pos,
            _scalar_type=int,
        )

    @property
    def plot(self) -> UniverseMappingPlotAccessor:
        return SeriesUniverseMappingPlotAccessor(self)

    @classmethod
    def from_vectors(
        cls, keys: Iterable[str], values: Iterable[T]
    ) -> SeriesUniverseMapping[T]:
        keys_tuple = tuple(keys)

        values_series = (
            pl.Series(values) if not isinstance(values, pl.Series) else values
        )

        return SeriesUniverseMapping.from_names_and_data(keys_tuple, values_series)


def _mapping_to_series[T: (float, int)](
    mapping: SeriesUniverseMapping,
    other_mapping: Mapping[Any, T] | VectorMapping[Any, T],
) -> pl.Series:
    idxs = (mapping.pos[k] for k in other_mapping)
    vals = (v for v in other_mapping.values())
    series = pl.zeros(len(mapping.names), eager=True)
    series.scatter(idxs, vals)
    return series
