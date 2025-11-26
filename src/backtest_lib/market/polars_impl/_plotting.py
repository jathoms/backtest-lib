from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Sequence,
    cast,
)

import altair as alt

alt.data_transformers.enable("vegafusion")

if TYPE_CHECKING:
    from backtest_lib.market.polars_impl._past_view import (
        PolarsByPeriod,
        PolarsBySecurity,
    )
    from backtest_lib.market.polars_impl._timeseries import PolarsTimeseries
    from backtest_lib.market.polars_impl._universe_mapping import SeriesUniverseMapping
    from backtest_lib.market.timeseries import Index
    from backtest_lib.universe import SecurityName


class SeriesUniverseMappingPlotAccessor:
    def __init__(self, obj: "SeriesUniverseMapping"):
        self._obj = obj
        self._series = obj.as_series()

    def __call__(
        self,
        *,
        kind: Literal["bar", "barh"] = "bar",
        **kwargs,
    ) -> Any:
        if kind == "bar":
            return self.bar(**kwargs)
        elif kind == "barh":
            return self.barh(**kwargs)
        raise ValueError(kind)

    def bar(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **kwargs,
    ) -> Any: ...

    def barh(
        self,
        select: int | slice | None = None,
        sort_by: Literal["value", "name"] = "value",
        ascending: bool = False,
        **kwargs,
    ) -> Any: ...

    def hist(
        self,
        bins: int | Iterable[float] = 20,
        select: int | slice | None = None,
        **kwargs,
    ) -> Any: ...


class PolarsTimeseriesPlotAccessor:
    def __init__(self, obj: "PolarsTimeseries"):
        self._obj = obj
        self._series = obj.as_series()

    def __call__(self, **kwargs) -> alt.Chart:
        return self.line(**kwargs)

    def line(
        self,
        y_padding: float = 0.01,
        color: str = "steelblue",
        smoothing: int = 1,
        **kwargs,
    ) -> alt.Chart:
        """Plot the series as a line chart."""

        datapoints = self._series.rolling_mean(window_size=smoothing)

        max_y = cast(float, datapoints.max()) * (1 + y_padding)
        min_y = cast(float, datapoints.min()) * (1 - y_padding)

        return (
            alt.Chart(
                datapoints.to_frame("value").with_columns(date=self._obj._axis.dt64)
            )
            .mark_line(tooltip=True, color=color)
            .encode(
                x="date",
                y=alt.Y(
                    "value",
                    scale=alt.Scale(
                        domain=[min_y, max_y],
                        clamp=True,
                    ),
                ),
                **kwargs,
            )
        )

    def bar(self, **kwargs) -> alt.Chart:
        """Plot the series as a bar chart"""
        return (
            alt.Chart(
                self._series.to_frame("value").with_columns(date=self._obj._axis.dt64)
            )
            .mark_bar(tooltip=True)
            .encode(x="date", y="value", **kwargs)
        )

    def hist(self, bins: int = 20, **kwargs) -> alt.Chart:
        """Histogram of the values."""

        return (
            alt.Chart(self._series.to_frame())
            .mark_bar(tooltip=True)
            .encode(
                x=alt.X(f"{self._series.name}:Q", bin=alt.Bin(maxbins=bins)),
                y="count()",
                **kwargs,
            )
            .interactive()
        )

    def kde(self, color="steelblue", **kwargs) -> alt.Chart:
        """KDE Plot of values."""
        return (
            alt.Chart(self._series.to_frame())
            .transform_density(self._series.name, as_=[self._series.name, "density"])
            .mark_area(
                stroke=color,
                strokeWidth=3,
                strokeOpacity=1,
                color=color,
                opacity=0.3,
                tooltip=True,
            )
            .encode(x=self._series.name, y="density:Q", **kwargs)
            .interactive()
        )


class PolarsByPeriodPlotAccessor:
    def __init__(self, obj: "PolarsByPeriod"):
        self._obj = obj
        self._df = self._obj.as_df()

    def __call__(self, **kwargs):
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        securities: Sequence[SecurityName] | None = None,
        **kwargs,
    ): ...

    def line(
        self,
        *,
        agg: Literal["mean", "median", "sum"],
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ): ...

    def box(
        self,
        *,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ): ...


class PolarsBySecurityPlotAccessor:
    def __init__(self, obj: "PolarsBySecurity"):
        self._obj = obj
        self._df = self._obj.as_df()

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def line(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        agg: Literal["none", "mean", "median", "sum"] = "none",
        facet: bool = False,
        max_securities: int | None = None,
        **kwargs,
    ): ...

    # - agg != "none": single aggregated line
    # - facet=True: one subplot per security (if small N)

    def heatmap(
        self,
        *,
        securities: Sequence[SecurityName] | None = None,
        periods: slice | Sequence[Index] | None = None,
        **kwargs,
    ): ...
