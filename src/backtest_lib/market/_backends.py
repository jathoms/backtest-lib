from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest_lib.market import PastView
    from backtest_lib.market.timeseries import Timeseries
    from backtest_lib.universe.universe_mapping import UniverseMapping


def _get_pastview_type_from_backend(backend: str) -> type[PastView]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import PolarsPastView

        return PolarsPastView
    raise ValueError(f"Could not find data backend {backend}")


def _get_mapping_type_from_backend(backend: str) -> type[UniverseMapping]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import PolarsUniverseMapping

        return PolarsUniverseMapping
    raise ValueError(f"Could not find data backend {backend}")


def _get_timeseries_type_from_backend(backend: str) -> type[Timeseries]:
    if backend == "polars":
        from backtest_lib.market.polars_impl import PolarsTimeseries

        return PolarsTimeseries
    raise ValueError(f"Could not find data backend {backend}")
