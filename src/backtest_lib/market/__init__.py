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

from backtest_lib.universe import (
    PastUniversePrices,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from backtest_lib.market.plotting import (
        ByPeriodPlotAccessor,
        BySecurityPlotAccessor,
        PastViewPlotAccessor,
    )
    from backtest_lib.market.timeseries import Comparable, Timeseries
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
    Sequence[VectorMapping[str, T]] | Sequence[Mapping[str, T]]
)


class SecurityAxisPolicy(Enum):
    """PLACEHOLDER"""

    STRICT = auto()
    SUBSET_OK = auto()
    SUPERSET_OK = auto()
    COERCE = auto()


class PeriodAxisPolicy(Enum):
    """PLACEHOLDER"""

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
    def periods(self) -> Sequence[Index]:
        """PLACEHOLDER PROPERTY"""
        ...

    @property
    @abstractmethod
    def securities(self) -> Sequence[str]:
        """PLACEHOLDER PROPERTY"""
        ...

    @property
    @abstractmethod
    def by_period(self) -> ByPeriod[ValueT, Index]:
        """PLACEHOLDER PROPERTY"""
        ...

    @property
    @abstractmethod
    def by_security(self) -> BySecurity[ValueT, Index]:
        """PLACEHOLDER PROPERTY"""
        ...

    @abstractmethod
    def between(
        self,
        start: Index | str,
        end: Index | str,
    ) -> Self:
        """PLACEHOLDER"""
        ...  # will not clone data, must be contiguous, performs a binary search

    @abstractmethod
    def after(
        self,
        start: Index | str,
        *,
        inclusive: bool = True,  # common expectation: include the start tick
    ) -> Self:
        """PLACEHOLDER"""
        ...

    @abstractmethod
    def before(
        self,
        end: Index | str,
        *,
        inclusive: bool = False,  # common expectation: half-open [.., end)
    ) -> Self:
        """PLACEHOLDER"""
        ...

    @staticmethod
    @abstractmethod
    def from_security_mappings(
        ms: SecurityMappings[Any],
        periods: Sequence[Index],
    ) -> Self:
        """PLACEHOLDER"""
        ...

    @staticmethod
    @abstractmethod
    def from_dataframe(df: pl.DataFrame | pd.DataFrame) -> Self:
        """PLACEHOLDER"""
        ...

    @property
    @abstractmethod
    def plot(self) -> PastViewPlotAccessor:
        """PLACEHOLDER"""
        ...


class ByPeriod[ValueT: (float, int), Index: Comparable](ABC):
    """PLACEHOLDER"""

    @abstractmethod
    def __len__(self) -> int:
        """PLACEHOLDER"""
        ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> UniverseMapping[ValueT]: ...

    @overload
    def __getitem__(self, key: slice) -> PastView[ValueT, Index]: ...

    @abstractmethod
    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> UniverseMapping[ValueT] | PastView[ValueT, Index]:
        """PLACEHOLDER"""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Index]:
        """PLACEHOLDER"""
        ...

    @property
    @abstractmethod
    def plot(self) -> ByPeriodPlotAccessor:
        """PLACEHOLDER"""
        ...

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
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        """PLACEHOLDER"""
        ...


class BySecurity[ValueT: (float, int), Index: Comparable](ABC):
    """PLACEHOLDER"""

    @abstractmethod
    def __len__(self) -> int:
        """PLACEHOLDER"""
        ...

    @overload
    def __getitem__(self, key: str) -> Timeseries[ValueT, Index]: ...

    @overload
    def __getitem__(self, key: Iterable[str]) -> PastView[ValueT, Index]: ...

    @abstractmethod
    def __getitem__(
        self, key: str | Iterable[str]
    ) -> Timeseries[ValueT, Index] | PastView[ValueT, Index]:
        """PLACEHOLDER"""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """PLACEHOLDER"""
        ...

    @property
    @abstractmethod
    def plot(self) -> BySecurityPlotAccessor:
        """PLACEHOLDER"""
        ...

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
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        """PLACEHOLDER"""
        ...


class MarketView[Index: Comparable]:
    """Holds a common set of signals for backtesting, as well as any custom signals
    defined by the user.

    Provides a mechanism for aligning, and enforcing the alignment of the
    :class:`~backtest_lib.market.PastView` for each signal.

    Type parameters:
        `Index`: A type that is comparable with itself. This will commonly be a
        datetime-like type such as ``np.datetime64``. This constraint allows
        us to enforce monotonicity on the market periods.

    Args:
        prices: A :class:`~backtest_lib.universe.PastUniversePrices` structure that
            holds close, open, high, and low prices for each period. Each price view is
            represented by a :class:`~backtest_lib.market.PastView`. All price views
            except for `close` are optional.
        tradable: An optional :class:`~backtest_lib.market.PastView` of tradability
            over the periods and securities of the reference view.
            volume: An optional :class:`~backtest_lib.market.PastView` of volume
            over the periods and securities of the reference view.
        signals: An optional mapping from :class:`str` to any custom
            :class:`~backtest_lib.market.PastView` specified by the user.
        security_policy: A :class:`~backtest_lib.market.SecurityAxisPolicy`
            determining how missing or misaligned securities are treated in the
            construction of the :class:`~backtest_lib.market.MarketView`.
        period_policy: A :class:`~backtest_lib.market.PeriodAxisPolicy`
            determining how missing periods are treated in the
            construction of the :class:`~backtest_lib.market.MarketView`.
        reference_view_for_axis_values: The :class:`~backtest_lib.market.PastView`
            of this market view used as a reference for the securities and periods
            for the market view as a whole. For example, when the reference view is
            "volume", a security or period in any other
            :class:`~backtest_lib.market.PastView` ("tradable", "prices.close", etc.)
            :class:`~backtest_lib.market.PastView` not found in the "volume"
            :class:`~backtest_lib.market.PastView` will be counted as `missing` and
            will trigger an exception when the associated axis policy is set to
            ``STRICT``.

            The reference view can be specified as an value of "signals" by passing
            "signal:<SIGNAL_NAME>" as the reference i.e "signal:carry" for a signal
            named "carry".
        backend: See ``backtest_lib.Backtest._backend``.


    Attributes:
        periods: The periods of the reference view. A :class:`~collections.abc.Sequence`
            of ``Index`` (See *Type parameters*).
        securities: The securities of the reference view. A
            :class:`~collections.abc.Sequence` of :class:`str`.

    """

    # TODO: polars has a FrameInitTypes type.
    # we might want to copy this idea and have a MarketViewInitTypes type
    # instead of just using Any here.
    def __init__(
        self,
        prices: PastUniversePrices[Index] | PastView[float, Index] | Any,
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

        periods = self._resolve_period_axis_spec(reference_view_for_axis_values)
        securities = self._resolve_security_axis_spec(reference_view_for_axis_values)

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
        ref_sec: Sequence[str],
        ref_periods: Sequence[Index],
    ) -> PastView:
        sec = view.securities
        if self._security_policy is SecurityAxisPolicy.STRICT:
            if len(sec) != len(ref_sec) or not all(
                a == b for a, b in zip(sec, ref_sec, strict=True)
            ):
                # TODO: improve this error message. will require more context in this
                # function i.e add a string of the name of the reference sequence
                raise ValueError("Securities must match reference exactly.")
            new_sec: Sequence[str] | None = None
        elif (
            self._security_policy is SecurityAxisPolicy.SUBSET_OK
            or self._security_policy is SecurityAxisPolicy.SUPERSET_OK
            or self._security_policy is SecurityAxisPolicy.COERCE
        ):
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
                a == b for a, b in zip(periods, ref_periods, strict=True)
            ):
                raise ValueError("Periods must match reference exactly.")
            new_per: Sequence[Index] | None = None
        elif (
            self._period_policy is PeriodAxisPolicy.INTERSECT
            or self._period_policy is PeriodAxisPolicy.FFILL
        ):
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

    def _resolve_security_axis_spec(self, spec: str) -> Sequence[str]:
        return self._resolve_axis_spec(spec).securities

    def _resolve_axis_spec(self, spec: str) -> PastView:
        view: PastView[int, Index] | PastView[float, Index] | None
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
            signals={k: v.by_period[:n_periods] for k, v in self.signals.items()},
        )

    def filter_securities(self, securities: Sequence[str]) -> Self:
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
            for k in self.signals
        }
        return MarketView(
            prices=self.prices.filter_securities(filtered_price),
            volume=self.volume.by_security[filtered_volume] if self.volume else None,
            tradable=(
                self.tradable.by_security[filtered_tradable] if self.tradable else None
            ),
            signals={
                k: v.by_security[filtered_signal_securities[k]]
                for k, v in self.signals.items()
            },
        )
