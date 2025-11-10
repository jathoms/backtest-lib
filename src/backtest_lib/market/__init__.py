from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import (
    Any,
    Protocol,
    Self,
    SupportsIndex,
    TypeVar,
    overload,
    runtime_checkable,
)

from backtest_lib.market.polars_impl import SecurityMappings
from backtest_lib.market.timeseries import Comparable, Timeseries
from backtest_lib.universe import (
    PastUniversePrices,
    PeriodIndex,
    SecurityName,
    UniverseMapping,
    UniverseVolume,
)
from backtest_lib.universe.vector_mapping import VectorMapping

Index = TypeVar("Index", bound=Comparable)

S = TypeVar(
    "S", bound=VectorMapping[SecurityName, Any], covariant=True
)  # mapping of securities to prices
P = TypeVar(
    "P", bound=Timeseries[Any, Comparable], covariant=True
)  # mapping of periods to some data (prices, volume, is_tradable)


class SecurityAxisPolicy(Enum):
    STRICT = auto()
    SUBSET_OK = auto()
    SUPERSET_OK = auto()
    COERCE = auto()


class PeriodAxisPolicy(Enum):
    STRICT = auto()
    INTERSECT = auto()
    FFILL = auto()


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
    def periods(self) -> Sequence[Index]: ...

    @property
    def securities(self) -> Sequence[SecurityName]: ...

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

    @staticmethod
    def from_security_mappings(
        ms: SecurityMappings[Any],
        periods: Sequence[Index],
    ) -> Self: ...


@runtime_checkable
class ByPeriod(Protocol[S, P, Index]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> S: ...

    @overload
    def __getitem__(self, key: slice) -> PastView[S, P, Index]: ...

    def __iter__(self) -> Iterator[Index]: ...


@runtime_checkable
class BySecurity(Protocol[S, P, Index]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SecurityName) -> P: ...

    @overload
    def __getitem__(self, key: Iterable[SecurityName]) -> PastView[S, P, Index]: ...

    def __iter__(self) -> Iterator[SecurityName]: ...


@dataclass(frozen=True)
class MarketView:
    prices: PastUniversePrices
    periods: Sequence[PeriodIndex]
    tradable: PastView[UniverseMapping[int], Timeseries, PeriodIndex] | None = None
    volume: PastView[UniverseVolume, Timeseries, PeriodIndex] | None = None
    signals: dict[str, PastView] = field(default_factory=dict)

    security_policy: SecurityAxisPolicy = SecurityAxisPolicy.STRICT
    period_axis_policy: PeriodAxisPolicy = PeriodAxisPolicy.STRICT
    reference_view_for_axis_values = "prices"

    def __post_init__(self):
        reference_period_values = self._resolve_period_axis_spec(
            self.reference_view_for_axis_values
        )
        reference_security_values = self._resolve_security_axis_spec(
            self.reference_view_for_axis_values
        )
        align = functools.partial(
            self._align,
            ref_sec=reference_security_values,
            ref_periods=reference_period_values,
        )
        new_prices = replace(
            self.prices,
            open=align(self.prices.close),
            close=align(self.prices.open) if self.prices.open is not None else None,
            high=align(self.prices.high) if self.prices.high is not None else None,
            low=align(self.prices.low) if self.prices.low is not None else None,
        )
        object.__setattr__(self, "prices", new_prices)
        if self.tradable is not None:
            object.__setattr__(self, "tradable", align(self.tradable))
        if self.volume is not None:
            object.__setattr__(self, "volume", align(self.volume))
        for name, view in self.signals.items():
            self.signals[name] = align(view)

    def _align(
        self,
        view: PastView,
        ref_sec: Sequence[SecurityName],
        ref_periods: Sequence[PeriodIndex],
    ) -> PastView:
        sec = view.securities
        if self.security_policy is SecurityAxisPolicy.STRICT:
            if len(sec) != len(ref_sec) or not all(
                a == b for a, b in zip(sec, ref_sec)
            ):
                # TODO: improve this error message. will require more context in this
                # function i.e add a string of the name of the reference sequence
                raise ValueError("Securities must match reference exactly.")
            new_sec: Sequence[SecurityName] | None = None
        elif self.security_policy is SecurityAxisPolicy.SUBSET_OK:
            raise NotImplementedError()
        elif self.security_policy is SecurityAxisPolicy.SUPERSET_OK:
            raise NotImplementedError()
        elif self.security_policy is SecurityAxisPolicy.COERCE:
            raise NotImplementedError()
        else:
            raise RuntimeError("Unknown security policy")

        periods = view.periods
        if self.period_axis_policy is PeriodAxisPolicy.STRICT:
            # TODO: would like a generalised vectorised equality check here,
            # we can't just do !=, as per/ref_periods can be an NDArray,
            # in which case the result is another NDArray of bools
            # that requires a numpy-specific function (.all()) to collapse
            # to a single bool
            if len(periods) != len(ref_periods) or not all(
                a == b for a, b in zip(periods, ref_periods)
            ):
                raise ValueError("Periods must match reference exactly.")
            new_per: Sequence[PeriodIndex] | None = None
        elif self.period_axis_policy is PeriodAxisPolicy.INTERSECT:
            raise NotImplementedError()
        elif self.period_axis_policy is PeriodAxisPolicy.FFILL:
            raise NotImplementedError()
        else:
            raise RuntimeError("Unknown period policy")

        if new_sec is None and new_per is None:
            return view

        raise NotImplementedError()
        # TODO: Add a reindexing method to the PastView protocol in some way
        # return view.reindex(securities=new_sec, periods=new_per)

    def _resolve_period_axis_spec(self, spec: str) -> Sequence[PeriodIndex]:
        return self._resolve_axis_spec(spec).periods

    def _resolve_security_axis_spec(self, spec: str) -> Sequence[SecurityName]:
        return self._resolve_axis_spec(spec).securities

    def _resolve_axis_spec(self, spec: str) -> PastView:
        if spec == "prices":
            # a bit of a sharp edge making "prices" exclusively tied to close prices.
            # TODO: loosen this requirement somewhat.
            view = self.prices.close
        elif spec == "tradable":
            view = self.tradable
        elif spec == "volume":
            view = self.volume
        elif spec.startswith("signal:"):
            key = spec.split(":", 1)[1]
            view = self.signals[key]
        else:
            raise ValueError(f"Unknown ref string: {spec}")
        if view is None:
            raise ValueError(f"Reference view '{spec}' is None for this MarketView")
        return view

    def truncated_to(self, n_periods: int) -> MarketView:
        return MarketView(
            prices=self.prices.truncated_to(n_periods),
            volume=self.volume.by_period[:n_periods] if self.volume else None,
            tradable=self.tradable.by_period[:n_periods] if self.tradable else None,
            periods=self.periods[:n_periods],
            signals={k: v.by_period[:n_periods] for k, v in self.signals.items()},
        )

    def filter_securities(self, securities: Sequence[SecurityName]) -> MarketView:
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
