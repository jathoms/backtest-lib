from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import altair as alt
import polars as pl

alt.data_transformers.enable("vegafusion")

from backtest_lib.market.plotting import (
    ByPeriodPlotAccessor,
    BySecurityPlotAccessor,
    PastViewPlotAccessor,
    TimeseriesPlotAccessor,
    UniverseMappingPlotAccessor,
)

if TYPE_CHECKING:
    from backtest_lib.market.polars_impl._past_view import (
        PolarsByPeriod,
        PolarsBySecurity,
        PolarsPastView,
    )
    from backtest_lib.market.polars_impl._timeseries import PolarsTimeseries
    from backtest_lib.market.polars_impl._universe_mapping import SeriesUniverseMapping


class PolarsPastViewPlotAccessor(PastViewPlotAccessor):
    def __init__(self, obj: "PolarsPastView"):
        self._obj = obj

    def __call__(self, kind: Literal["bar", "line"] = "line", **kwargs):
        if kind == "bar":
            return self.bar(**kwargs)
        elif kind == "line":
            return self.line(**kwargs)
        else:
            raise ValueError(kind)

    def bar(
        self,
        **kwargs,
    ) -> alt.LayerChart:
        return self._obj.by_period.plot.bar()

    def line(
        self,
        agg: Literal["none", "mean", "sum"] = "none",
        y_padding: float = 0.01,
        smoothing: int = 1,
        **kwargs,
    ) -> alt.LayerChart:
        return self._obj.by_security.plot.line(agg, y_padding, smoothing)


class SeriesUniverseMappingPlotAccessor(UniverseMappingPlotAccessor):
    def __init__(self, obj: "SeriesUniverseMapping"):
        self._obj = obj
        self._series = obj.as_series()

    def __call__(
        self,
        kind: Literal["bar", "barh"] = "bar",
        **kwargs,
    ) -> alt.Chart:
        if kind == "bar":
            return self.bar(**kwargs)
        elif kind == "barh":
            return self.barh(**kwargs)
        raise ValueError(kind)

    def bar(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        color: str = "steelblue",
        **kwargs,
    ) -> alt.Chart:
        frame = self._series.to_frame("value").with_columns(
            name=pl.Series(self._obj.names)
        )

        if sort_by != "none":
            sort = alt.SortField(
                field=sort_by, order="descending" if descending else "ascending"
            )
            frame = frame.sort(sort_by, descending=descending)
        else:
            sort = None

        if top is not None:
            frame = frame.slice(0, top)

        return (
            alt.Chart(frame)
            .mark_bar(tooltip=True, color=color)
            .encode(x=alt.X("name:N", sort=sort), y="value:Q", **kwargs)
        )

    def barh(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        color: str = "steelblue",
        **kwargs,
    ) -> alt.Chart:
        frame = self._series.to_frame("value").with_columns(
            name=pl.Series(self._obj.names)
        )

        if sort_by != "none":
            sort = alt.SortField(
                field=sort_by, order="descending" if descending else "ascending"
            )
            frame = frame.sort(sort_by, descending=descending)
        else:
            sort = None

        if top is not None:
            frame = frame.slice(0, top)

        return (
            alt.Chart(frame)
            .mark_bar(tooltip=True, color=color)
            .encode(x="value:Q", y=alt.Y("name:N", sort=sort), **kwargs)
        )

    def stacked_bar(
        self,
        top: int | None = None,
        sort_by: Literal["value", "name", "none"] = "value",
        descending: bool = True,
        bar_label: str = "",
        bar_label_encoding_type: str = "N",
        **kwargs,
    ) -> alt.Chart:
        frame = self._series.to_frame("value").with_columns(
            name=pl.Series(self._obj.names), bar_label=pl.lit(bar_label)
        )

        if sort_by != "none":
            frame = frame.sort(sort_by, descending=descending)

        if top is not None:
            frame = frame.slice(0, top)

        return (
            alt.Chart(frame)
            .mark_bar(tooltip=True)
            .encode(
                x=alt.X(f"bar_label:{bar_label_encoding_type}"),
                y=alt.Y("value:Q", stack="zero"),
                color="name:N",
                **kwargs,
            )
        )


class PolarsTimeseriesPlotAccessor(TimeseriesPlotAccessor):
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
                x="date:T",
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
            .encode(x="date:T", y="value", **kwargs)
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


class PolarsByPeriodPlotAccessor(ByPeriodPlotAccessor):
    def __init__(self, obj: "PolarsByPeriod"):
        self._obj = obj

    def __call__(self, **kwargs):
        return self.bar(**kwargs)

    def bar(
        self,
        **kwargs,
    ) -> alt.LayerChart:
        charts = [
            self._obj[idx].plot.stacked_bar(
                bar_label=period, bar_label_encoding_type="T"
            )
            for idx, period in enumerate(self._obj)
        ]
        return alt.layer(*charts, **kwargs)


class PolarsBySecurityPlotAccessor(BySecurityPlotAccessor):
    def __init__(self, obj: "PolarsBySecurity"):
        self._obj = obj
        self._df = obj.as_df(show_periods=True, lazy=False)

    def __call__(self, **kwargs):
        return self.line(**kwargs)

    def line(
        self,
        agg: Literal["none", "mean", "sum"] = "none",
        y_padding: float = 0.01,
        smoothing: int = 1,
        **kwargs,
    ):
        value_cols = [
            c
            for c, dt in zip(self._df.columns, self._df.dtypes)
            if dt.is_numeric() and c != "date"
        ]
        df = self._df.select(
            "date", pl.all().exclude("date").rolling_mean(window_size=smoothing)
        )
        if agg == "none":
            df_long = df.unpivot(
                index=["date"],
                on=value_cols,
                variable_name="series",
                value_name="value",
            )

            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("date:T"),
                    y=alt.Y("value:Q"),
                    color=alt.Color("series:N"),
                    tooltip=[
                        alt.Tooltip("series:N"),
                        alt.Tooltip("value:Q"),
                        alt.Tooltip("date:T"),
                    ],
                )
            )
            return chart

        if agg == "mean":
            agg_expr = pl.mean_horizontal()
        elif agg == "sum":
            agg_expr = pl.sum_horizontal()
        else:
            raise ValueError(f"Unknown agg type '{agg}'")

        agg_df = (
            df.lazy()
            .with_columns(agg_expr.alias(agg))
            .select(pl.col("date"), pl.col(agg))
            .collect()
        )
        max_y = cast(float, agg_df[agg].max()) * (1 + y_padding)
        min_y = cast(float, agg_df[agg].min()) * (1 - y_padding)
        return (
            alt.Chart(agg_df)
            .mark_line(tooltip=True)
            .encode(
                x="date:T",
                y=alt.Y(
                    f"{agg}:Q",
                    scale=alt.Scale(
                        domain=[min_y, max_y],
                        clamp=True,
                    ),
                ),
                **kwargs,
            )
        )
