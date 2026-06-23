from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast, overload

import polars as pl

from backtest_lib._rs import PyUniverseMapping
from backtest_lib.market.plotting import UniverseMappingPlotAccessor
from backtest_lib.market.polars_impl._axis import SecurityAxis
from backtest_lib.market.polars_impl._universe_mapping import PolarsUniverseMapping
from backtest_lib.universe import Universe
from backtest_lib.universe.universe_mapping import UniverseMapping
from backtest_lib.universe.vector_mapping import VectorMapping

T = TypeVar("T", int, float)


@overload
def _wrap_polars_mapping(
    mapping: PolarsUniverseMapping[int],
) -> NativeUniverseMapping[int]: ...


@overload
def _wrap_polars_mapping(
    mapping: PolarsUniverseMapping[float],
) -> NativeUniverseMapping[float]: ...


def _wrap_polars_mapping(
    mapping: PolarsUniverseMapping[int] | PolarsUniverseMapping[float],
) -> NativeUniverseMapping[int] | NativeUniverseMapping[float]:
    return cast(
        NativeUniverseMapping[int] | NativeUniverseMapping[float],
        NativeUniverseMapping.from_polars_mapping(cast(Any, mapping)),
    )


@dataclass(frozen=True, init=False)
class NativeUniverseMapping[T: (float, int)](UniverseMapping[T]):
    _inner: PolarsUniverseMapping[T] = field(repr=False)
    _m: PyUniverseMapping = field(repr=False)

    def __init__(self, inner: PolarsUniverseMapping[T], _m: PyUniverseMapping):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_m", _m)

    @classmethod
    def _from_rust_mapping(
        cls, axis: SecurityAxis, mapping: PyUniverseMapping
    ) -> NativeUniverseMapping[float]:
        inner = PolarsUniverseMapping(axis, cast(pl.Series, mapping.values()), float)
        return cls(inner, mapping)

    @staticmethod
    def _coerce_rust_operand(other: Any) -> PyUniverseMapping | float:
        if isinstance(other, NativeUniverseMapping):
            return other._m
        if isinstance(other, Mapping):
            return NativeUniverseMapping.from_vectors(other.keys(), other.values())._m
        return float(other)

    @classmethod
    def from_polars_mapping(
        cls, mapping: PolarsUniverseMapping[T]
    ) -> NativeUniverseMapping[T]:
        return cls(
            mapping,
            PyUniverseMapping.from_series(mapping.axis.native, mapping.to_series()),
        )

    @classmethod
    def from_names_and_data(
        cls,
        names: Universe,
        data: pl.Series,
        dtype: type[T] | None = None,
    ) -> NativeUniverseMapping[T]:
        return cls.from_polars_mapping(
            cast(
                Any,
                PolarsUniverseMapping.from_names_and_data(
                    names, data, cast(Any, dtype)
                ),
            )
        )

    @classmethod
    @overload
    def from_vectors(
        cls, keys: Iterable[str], values: Iterable[int]
    ) -> NativeUniverseMapping[int]: ...

    @classmethod
    @overload
    def from_vectors(
        cls, keys: Iterable[str], values: Iterable[float]
    ) -> NativeUniverseMapping[float]: ...

    @classmethod
    @overload
    def from_vectors(
        cls, keys: Iterable[str], values: Iterable[int | float]
    ) -> NativeUniverseMapping[float]: ...

    @classmethod
    def from_vectors(
        cls,
        keys: Iterable[str],
        values: Iterable[int | float],
    ) -> NativeUniverseMapping[int] | NativeUniverseMapping[float]:
        return _wrap_polars_mapping(PolarsUniverseMapping.from_vectors(keys, values))

    @property
    def names(self) -> Universe:
        return tuple(self._m.keys())

    @property
    def plot(self) -> UniverseMappingPlotAccessor:
        return self._inner.plot

    def to_series(self) -> pl.Series:
        return cast(pl.Series, self._m.values())

    @overload
    def __getitem__(self, key: str) -> T: ...

    @overload
    def __getitem__(self, key: Iterable[str]) -> pl.Series: ...

    def __getitem__(self, key: str | Iterable[str]) -> T | pl.Series:
        try:
            return cast(T | pl.Series, self._m[key])
        except TypeError as e:
            raise ValueError(f"Unsupported index '{key}' with type {type(key)}") from e

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def __add__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        return self._from_rust_mapping(self._inner.axis, self._m.__add__(operand))

    def __radd__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        axis = (
            other._inner.axis
            if isinstance(other, NativeUniverseMapping)
            else self._inner.axis
        )
        return self._from_rust_mapping(axis, self._m.__radd__(operand))

    def __sub__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        return self._from_rust_mapping(self._inner.axis, self._m.__sub__(operand))

    def __rsub__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        axis = (
            other._inner.axis
            if isinstance(other, NativeUniverseMapping)
            else self._inner.axis
        )
        return self._from_rust_mapping(axis, self._m.__rsub__(operand))

    def __mul__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        return self._from_rust_mapping(self._inner.axis, self._m.__mul__(operand))

    def __rmul__(self, other) -> NativeUniverseMapping:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        axis = (
            other._inner.axis
            if isinstance(other, NativeUniverseMapping)
            else self._inner.axis
        )
        return self._from_rust_mapping(axis, self._m.__rmul__(operand))

    def __truediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> NativeUniverseMapping[float]:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        return self._from_rust_mapping(self._inner.axis, self._m.__truediv__(operand))

    def __rtruediv__(
        self,
        other: VectorMapping | float | int | Mapping[str, int | float],
    ) -> NativeUniverseMapping[float]:
        try:
            operand = self._coerce_rust_operand(other)
        except (TypeError, ValueError):
            return cast(Any, NotImplemented)
        return self._from_rust_mapping(
            other._inner.axis
            if isinstance(other, NativeUniverseMapping)
            else self._inner.axis,
            self._m.__rtruediv__(operand),
        )

    def sum(self) -> T:
        return cast(T, self._m.sum())

    def abs(self) -> NativeUniverseMapping[T]:
        return _wrap_polars_mapping(self._inner.abs())

    def truncate(self) -> NativeUniverseMapping[int]:
        return _wrap_polars_mapping(self._inner.truncate())

    def floor(self) -> NativeUniverseMapping[int]:
        return _wrap_polars_mapping(self._inner.floor())
