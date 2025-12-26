from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    SupportsIndex,
    overload,
)

import pandas as pd
import polars as pl

from backtest_lib.universe import (
    PastUniversePrices,
)

if TYPE_CHECKING:
    from backtest_lib.market.plotting import (
        ByPeriodPlotAccessor,
        BySecurityPlotAccessor,
        PastViewPlotAccessor,
    )
    from backtest_lib.market.timeseries import Comparable, Timeseries
    from backtest_lib.universe import (
        SecurityName,
    )
    from backtest_lib.universe.universe_mapping import UniverseMapping
    from backtest_lib.universe.vector_mapping import VectorMapping


def get_pastview_from_mapping(backend: str) -> type[PastView]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import PolarsPastView

        return PolarsPastView
    else:
        raise ValueError(f"Could not find data backend {backend}")


def get_mapping_type_from_mapping(backend: str) -> type[UniverseMapping]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import SeriesUniverseMapping

        return SeriesUniverseMapping
    else:
        raise ValueError(f"Could not find data backend {backend}")


def get_timeseries_type_from_mapping(backend: str) -> type[Timeseries]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import PolarsTimeseries

        return PolarsTimeseries
    else:
        raise ValueError(f"Could not find data backend {backend}")


type SecurityMappings[T: (float, int)] = (
    Sequence[VectorMapping[SecurityName, T]] | Sequence[Mapping[SecurityName, T]]
)


class SecurityAxisPolicy(Enum):
    STRICT = auto()
    SUBSET_OK = auto()
    SUPERSET_OK = auto()
    COERCE = auto()


class PeriodAxisPolicy(Enum):
    STRICT = auto()
    INTERSECT = auto()
    FFILL = auto()


class PastView[ValueT: (float, int), Index: Comparable](ABC):
    """
    Time-fenced read-only series up to the current decision point.

    This protocol can abstract over different implementations (list-backed,
    NumPy array-backed, mmap, etc.) that present a "fenced" slice of
    historical snapshots ending at "now", with no lookahead as to
    reduce the risk of lookahead bias while maintaining an ergonomic
    interface to access the market conditions.
    """

    @property
    @abstractmethod
    def periods(self) -> Sequence[Index]: ...

    @property
    @abstractmethod
    def securities(self) -> Sequence[SecurityName]: ...

    @property
    @abstractmethod
    def by_period(self) -> ByPeriod[ValueT, Index]: ...

    @property
    @abstractmethod
    def by_security(self) -> BySecurity[ValueT, Index]: ...

    @abstractmethod
    def between(
        self,
        start: Index | str,
        end: Index | str,
    ) -> Self: ...  # will not clone data, must be contiguous, performs a binary search

    @abstractmethod
    def after(
        self,
        start: Index | str,
        *,
        inclusive: bool = True,  # common expectation: include the start tick
    ) -> Self: ...

    @abstractmethod
    def before(
        self,
        end: Index | str,
        *,
        inclusive: bool = False,  # common expectation: half-open [.., end)
    ) -> Self: ...

    @staticmethod
    @abstractmethod
    def from_security_mappings(
        ms: SecurityMappings[Any],
        periods: Sequence[Index],
    ) -> Self: ...

    @staticmethod
    @abstractmethod
    def from_dataframe(df: pl.DataFrame | pd.DataFrame) -> Self: ...

    @property
    @abstractmethod
    def plot(self) -> PastViewPlotAccessor: ...


class ByPeriod[ValueT: (float, int), Index: Comparable](ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> UniverseMapping[ValueT]: ...

    @overload
    def __getitem__(self, key: slice) -> PastView[ValueT, Index]: ...

    @abstractmethod
    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> UniverseMapping[ValueT] | PastView[ValueT, Index]: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Index]: ...

    @property
    @abstractmethod
    def plot(self) -> ByPeriodPlotAccessor: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[False] = False,
        backend: Literal["polars"],
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[False] = False,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: Literal[True],
        backend: Literal["polars"],
    ) -> pl.LazyFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_securities: bool = ...,
        lazy: bool = ...,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    @abstractmethod
    def to_dataframe(
        self,
        *,
        show_securities: bool = False,
        lazy: bool = False,
        backend: Literal["polars", "pandas"] = "polars",
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame: ...


class BySecurity[ValueT: (float, int), Index: Comparable](ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SecurityName) -> Timeseries[ValueT, Index]: ...

    @overload
    def __getitem__(self, key: Iterable[SecurityName]) -> PastView[ValueT, Index]: ...

    @abstractmethod
    def __getitem__(
        self, key: SecurityName | Iterable[SecurityName]
    ) -> Timeseries[ValueT, Index] | PastView[ValueT, Index]: ...

    def __iter__(self) -> Iterator[SecurityName]: ...

    @property
    @abstractmethod
    def plot(self) -> BySecurityPlotAccessor: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: Literal[False] = ...,
        backend: Literal["polars"] = ...,
    ) -> pl.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: Literal[True],
        backend: Literal["polars"] = ...,
    ) -> pl.LazyFrame: ...

    @overload
    def to_dataframe(
        self,
        *,
        show_periods: bool = ...,
        lazy: bool = ...,
        backend: Literal["pandas"],
    ) -> pd.DataFrame: ...

    @abstractmethod
    def to_dataframe(
        self,
        *,
        show_periods: bool = True,
        lazy: bool = False,
        backend: Literal["polars", "pandas"] = "polars",
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame: ...


class MarketView[Index: Comparable]:
    # TODO: polars has a FrameInitTypes type.
    # we might want to copy this idea and have a MarketViewInitTypes type
    # instead of just using Any here.
    def __init__(
        self,
        prices: PastUniversePrices[Index] | PastView[float, Index] | Any,
        periods: Sequence[Index] | None = None,
        securities: Sequence[str] | None = None,
        tradable: PastView[int, Index] | Any | None = None,
        volume: PastView[int, Index] | Any | None = None,
        signals: dict[str, PastView[Any, Index] | Any] | None = None,
        security_policy: SecurityAxisPolicy = SecurityAxisPolicy.STRICT,
        period_policy: PeriodAxisPolicy = PeriodAxisPolicy.STRICT,
        reference_view_for_axis_values: str = "prices",
        backend: str = "polars",
    ):
        backend_pastview_type = get_pastview_from_mapping(backend)

        if not isinstance(prices, PastUniversePrices):
            if isinstance(prices, PastView):
                prices = PastUniversePrices(close=prices)
            else:
                # assume the user passes close prices
                prices = PastUniversePrices(
                    close=backend_pastview_type.from_dataframe(prices)
                )

        if tradable is not None and not isinstance(tradable, PastView):
            tradable = backend_pastview_type.from_dataframe(tradable)

        if volume is not None and not isinstance(volume, PastView):
            volume = backend_pastview_type.from_dataframe(volume)

        if signals is not None:
            normalised_signals: dict[str, PastView[Any, Index]] = {
                signal: (
                    backend_pastview_type.from_dataframe(data)
                    if not isinstance(data, PastView)
                    else data
                )
                for signal, data in signals.items()
            }
        else:
            normalised_signals = {}

        self._prices: PastUniversePrices[Index] = prices
        self._tradable: PastView[int, Index] | None = tradable
        self._volume: PastView[int, Index] | None = volume
        self._signals: dict[str, PastView[Any, Index]] = normalised_signals

        if periods is None:
            periods = self._resolve_period_axis_spec(reference_view_for_axis_values)

        if securities is None:
            securities = self._resolve_security_axis_spec(
                reference_view_for_axis_values
            )

        self._periods: Sequence[Index] = periods
        self._securities: Sequence[str] = securities
        self._security_policy: SecurityAxisPolicy = security_policy
        self._period_policy: PeriodAxisPolicy = period_policy
        self._backend: str = backend

        self._check_axis_alignment(reference_view_for_axis_values)

    @property
    def prices(self) -> PastUniversePrices[Index]:
        return self._prices

    @property
    def periods(self) -> Sequence[Index]:
        return self._periods

    @property
    def securities(self) -> Sequence[str]:
        return self._securities

    @property
    def tradable(self) -> PastView[int, Index] | None:
        return self._tradable

    @property
    def volume(self) -> PastView[int, Index] | None:
        return self._volume

    @property
    def signals(self) -> dict[str, PastView[Any, Index]]:
        return self._signals

    def _check_axis_alignment(self, reference_view_for_axis_values: str) -> None:
        reference_period_values = self._resolve_period_axis_spec(
            reference_view_for_axis_values
        )
        reference_security_values = self._resolve_security_axis_spec(
            reference_view_for_axis_values
        )

        align = functools.partial(
            self._align,
            ref_sec=reference_security_values,
            ref_periods=reference_period_values,
        )

        new_prices = replace(
            self._prices,
            close=align(self._prices.close),
            open=align(self._prices.open) if self._prices.open is not None else None,
            high=align(self._prices.high) if self._prices.high is not None else None,
            low=align(self._prices.low) if self._prices.low is not None else None,
        )
        self._prices = new_prices

        if self._tradable is not None:
            self._tradable = align(self._tradable)
        if self._volume is not None:
            self._volume = align(self._volume)

        for name, view in self._signals.items():
            self._signals[name] = align(view)

    def _align(
        self,
        view: PastView,
        ref_sec: Sequence[SecurityName],
        ref_periods: Sequence[Index],
    ) -> PastView:
        sec = view.securities
        if self._security_policy is SecurityAxisPolicy.STRICT:
            if len(sec) != len(ref_sec) or not all(
                a == b for a, b in zip(sec, ref_sec)
            ):
                # TODO: improve this error message. will require more context in this
                # function i.e add a string of the name of the reference sequence
                raise ValueError("Securities must match reference exactly.")
            new_sec: Sequence[SecurityName] | None = None
        elif self._security_policy is SecurityAxisPolicy.SUBSET_OK or self._security_policy is SecurityAxisPolicy.SUPERSET_OK or self._security_policy is SecurityAxisPolicy.COERCE:
            raise NotImplementedError()
        else:
            raise RuntimeError("Unknown security policy")

        periods = view.periods
        if self._period_policy is PeriodAxisPolicy.STRICT:
            # TODO: would like a generalised vectorised equality check here,
            # we can't just do !=, as per/ref_periods can be an NDArray,
            # in which case the result is another NDArray of bools
            # that requires a numpy-specific function (.all()) to collapse
            # to a single bool
            if len(periods) != len(ref_periods) or not all(
                a == b for a, b in zip(periods, ref_periods)
            ):
                raise ValueError("Periods must match reference exactly.")
            new_per: Sequence[Index] | None = None
        elif self._period_policy is PeriodAxisPolicy.INTERSECT or self._period_policy is PeriodAxisPolicy.FFILL:
            raise NotImplementedError()
        else:
            raise RuntimeError("Unknown period policy")

        if new_sec is None and new_per is None:
            return view

        raise NotImplementedError()
        # TODO: Add a reindexing method to the PastView protocol in some way
        # return view.reindex(securities=new_sec, periods=new_per)

    def _resolve_period_axis_spec(self, spec: str) -> Sequence[Index]:
        return self._resolve_axis_spec(spec).periods

    def _resolve_security_axis_spec(self, spec: str) -> Sequence[SecurityName]:
        return self._resolve_axis_spec(spec).securities

    def _resolve_axis_spec(self, spec: str) -> PastView:
        if spec == "prices":
            # a bit of a sharp edge making "prices" exclusively tied to close prices.
            # TODO: loosen this requirement somewhat.
            view = self._prices.close
        elif spec == "tradable":
            view = self._tradable
        elif spec == "volume":
            view = self._volume
        elif spec.startswith("signal:"):
            key = spec.split(":", 1)[1]
            view = self._signals[key]
        else:
            raise ValueError(f"Unknown ref string: {spec}")
        if view is None:
            raise ValueError(f"Reference view '{spec}' is None for this MarketView")
        return view

    def truncated_to(self, n_periods: int) -> Self:
        return MarketView(
            prices=self.prices.truncated_to(n_periods),
            volume=self.volume.by_period[:n_periods] if self.volume else None,
            tradable=self.tradable.by_period[:n_periods] if self.tradable else None,
            periods=self.periods[:n_periods],
            signals={k: v.by_period[:n_periods] for k, v in self.signals.items()},
        )

    def filter_securities(self, securities: Sequence[SecurityName]) -> Self:
        filtered_price = [
            sec for sec in securities if sec in self.prices.close.securities
        ]
        filtered_volume = (
            [sec for sec in securities if sec in self.volume.securities]
            if self.volume is not None
            else []
        )
        filtered_tradable = (
            [sec for sec in securities if sec in self.tradable.securities]
            if self.tradable is not None
            else []
        )
        filtered_signal_securities = {
            k: [sec for sec in securities if sec in self.signals[k].securities]
            for k in self.signals.keys()
        }
        return MarketView(
            prices=self.prices.filter_securities(filtered_price),
            volume=self.volume.by_security[filtered_volume] if self.volume else None,
            tradable=self.tradable.by_security[filtered_tradable]
            if self.tradable
            else None,
            periods=self.periods,
            signals={
                k: v.by_security[filtered_signal_securities[k]]
                for k, v in self.signals.items()
            },
        )
