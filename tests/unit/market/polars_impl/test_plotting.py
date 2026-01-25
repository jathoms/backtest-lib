import altair as alt
import numpy as np
import polars as pl
import pytest

from backtest_lib.market.polars_impl import (
    PolarsPastView,
    PolarsTimeseries,
    PolarsUniverseMapping,
)

# This data transformer, though different from the production code,
# enables a nice to_dict call ("vegafusion" makes to_dict a little painful).
alt.data_transformers.enable("default")


@pytest.fixture()
def small_timeseries() -> PolarsTimeseries:
    return PolarsTimeseries.from_vectors(
        values=[1.0, 2.0, 3.0],
        periods=[
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
        ],
        name="value",
    )


@pytest.fixture()
def small_past_view() -> PolarsPastView:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [1.0, 2.0],
            "BBB": [3.0, 4.0],
        }
    )
    return PolarsPastView.from_dataframe(df)


@pytest.fixture()
def small_mapping() -> PolarsUniverseMapping:
    return PolarsUniverseMapping.from_vectors(["AAA", "BBB", "CCC"], [3, 1, 2])


def test_past_view_plot_call_dispatch(small_past_view: PolarsPastView) -> None:
    chart = small_past_view.plot(kind="bar")
    assert isinstance(chart, alt.LayerChart)

    chart = small_past_view.plot(kind="line", agg="mean")
    spec = chart.to_dict()  # type: ignore[union-attr]
    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["y"]["field"] == "mean"


def test_past_view_plot_invalid_kind(small_past_view: PolarsPastView) -> None:
    with pytest.raises(ValueError):
        small_past_view.plot(kind="scatter")  # type: ignore[arg-type]


def test_universe_mapping_bar_sort_top(small_mapping: PolarsUniverseMapping) -> None:
    chart = small_mapping.plot.bar(
        top=2, sort_by="value", descending=False, color="red"
    )
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["mark"]["color"] == "red"
    assert spec["encoding"]["x"]["field"] == "name"
    assert spec["encoding"]["x"]["sort"]["order"] == "ascending"
    assert chart.data.height == 2


def test_universe_mapping_invalid_kind(small_mapping: PolarsUniverseMapping) -> None:
    with pytest.raises(ValueError):
        small_mapping.plot(kind="stack")  # type: ignore[arg-type]


def test_universe_mapping_barh_sort_none(small_mapping: PolarsUniverseMapping) -> None:
    chart = small_mapping.plot.barh(sort_by="none")
    spec = chart.to_dict()
    assert spec["encoding"]["x"]["field"] == "value"
    assert spec["encoding"]["y"]["field"] == "name"
    assert spec["encoding"]["y"]["sort"] is None


def test_universe_mapping_stacked_bar(small_mapping: PolarsUniverseMapping) -> None:
    chart = small_mapping.plot.stacked_bar(bar_label="All", bar_label_encoding_type="N")
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "bar_label"
    assert spec["encoding"]["y"]["stack"] == "zero"
    assert spec["encoding"]["color"]["field"] == "name"


def test_timeseries_plot_call_returns_line(small_timeseries: PolarsTimeseries) -> None:
    chart = small_timeseries.plot()
    assert isinstance(chart, alt.Chart)
    assert chart.to_dict()["mark"]["type"] == "line"


def test_timeseries_line_domain_and_smoothing(
    small_timeseries: PolarsTimeseries,
) -> None:
    chart = small_timeseries.plot.line(y_padding=0.1, smoothing=2)
    spec = chart.to_dict()
    domain = spec["encoding"]["y"]["scale"]["domain"]
    assert domain[0] == pytest.approx(1.35)
    assert domain[1] == pytest.approx(2.75)
    assert chart.data["value"].to_list()[0] is None


def test_timeseries_bar_plot(small_timeseries: PolarsTimeseries) -> None:
    chart = small_timeseries.plot.bar()
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "date"
    assert spec["encoding"]["y"]["field"] == "value"


def test_timeseries_bar_plot_data(small_timeseries: PolarsTimeseries) -> None:
    chart = small_timeseries.plot.bar()
    data = chart.data
    assert data["value"].to_list() == [1.0, 2.0, 3.0]
    expected_dates = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        dtype="datetime64[us]",
    )
    np.testing.assert_array_equal(data["date"].to_numpy(), expected_dates)


def test_timeseries_hist_plot(small_timeseries: PolarsTimeseries) -> None:
    chart = small_timeseries.plot.hist(bins=7)
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["bin"]["maxbins"] == 7
    assert spec["encoding"]["y"]["aggregate"] == "count"


def test_timeseries_kde_plot(small_timeseries: PolarsTimeseries) -> None:
    chart = small_timeseries.plot.kde(color="orange")
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "area"
    assert spec["mark"]["color"] == "orange"
    assert spec["transform"][0]["density"] == "value"


def test_by_period_bar_layer_count(small_past_view: PolarsPastView) -> None:
    chart = small_past_view.by_period.plot.bar()
    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == len(small_past_view.periods)


def test_by_security_line_agg_none(small_past_view: PolarsPastView) -> None:
    chart = small_past_view.by_security.plot.line(agg="none")
    spec = chart.to_dict()
    assert spec["encoding"]["color"]["field"] == "series"
    assert len(spec["encoding"]["tooltip"]) == 3


def test_by_security_line_agg_mean(small_past_view: PolarsPastView) -> None:
    chart = small_past_view.by_security.plot.line(
        agg="mean", y_padding=0.0, smoothing=1
    )
    spec = chart.to_dict()
    assert spec["encoding"]["y"]["field"] == "mean"


def test_by_security_line_invalid_agg(small_past_view: PolarsPastView) -> None:
    with pytest.raises(ValueError):
        small_past_view.by_security.plot.line(agg="median")  # type: ignore[arg-type]
