import decimal

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtest_lib.market.polars_impl import PolarsPastView


@pytest.fixture()
def small_past_view() -> PolarsPastView:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [10.0, 20.0, 30.0],
        }
    )
    return PolarsPastView.from_dataframe(df)


def test_from_security_mappings_empty() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings([], [])


def test_from_security_mappings_empty_period_mapping() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings([{}], [np.datetime64("2024-01-01")])


def test_from_security_mappings_period_mismatch() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings(
            [{"AAA": 1.0}],
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        )


def test_from_security_mappings_key_mismatch() -> None:
    with pytest.raises(KeyError):
        PolarsPastView.from_security_mappings(
            [{"AAA": 1.0, "BBB": 2.0}, {"AAA": 1.5, "CCC": 3.0}],
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        )


def test_from_security_mappings_mixed_types() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings(
            [{"AAA": 1}, {"AAA": 1.5}],
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        )


def test_from_security_mappings_unsupported_type() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings(
            [{"AAA": "x"}, {"AAA": "y"}],
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        )


def test_from_security_mappings_success() -> None:
    periods = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    pv = PolarsPastView.from_security_mappings(
        [{"AAA": 1.0, "BBB": 2.0}, {"AAA": 1.5, "BBB": 2.5}],
        periods,
    )

    assert pv.securities == ("AAA", "BBB")
    assert pv.periods == tuple(np.array(periods, dtype="datetime64[us]"))
    assert pv.by_security["AAA"].to_series().to_list() == [1.0, 1.5]


def test_from_dataframe_string_dates() -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [1.0, 2.0],
            "BBB": [3.0, 4.0],
        }
    )
    pv = PolarsPastView.from_dataframe(df)
    expected = tuple(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[us]"))
    assert pv.periods == expected
    assert pv.securities == ("AAA", "BBB")


def test_from_dataframe_missing_date() -> None:
    df = pl.DataFrame({"AAA": [1.0], "BBB": [2.0]})
    with pytest.raises(ValueError):
        PolarsPastView.from_dataframe(df)


def test_from_dataframe_invalid_date_dtype() -> None:
    df = pl.DataFrame({"date": [["2024-01-01"], ["2024-01-02"]], "AAA": [1.0, 2.0]})
    with pytest.raises(ValueError):
        PolarsPastView.from_dataframe(df)


def test_from_dataframe_invalid_object() -> None:
    with pytest.raises(ValueError):
        PolarsPastView.from_dataframe(object())  # type: ignore[arg-type]


def test_by_security_single_series(small_past_view: PolarsPastView) -> None:
    series = small_past_view.by_security["AAA"]
    assert series.to_series().to_list() == [1.0, 2.0, 3.0]
    assert len(series) == 3


def test_by_security_selection_returns_past_view(
    small_past_view: PolarsPastView,
) -> None:
    subset = small_past_view.by_security[["BBB", "AAA"]]
    assert subset.securities == ("BBB", "AAA")
    assert subset.by_security["BBB"].to_series().to_list() == [10.0, 20.0, 30.0]
    assert subset.by_security["AAA"].to_series().to_list() == [1.0, 2.0, 3.0]


def test_by_security_selection_unknown_key(small_past_view: PolarsPastView) -> None:
    with pytest.raises(KeyError):
        small_past_view.by_security["CCC"]


def test_by_period_out_of_range(small_past_view: PolarsPastView) -> None:
    with pytest.raises(IndexError):
        small_past_view.by_period[99]
    with pytest.raises(IndexError):
        small_past_view.by_period[-99]


def test_by_period_single_mapping(small_past_view: PolarsPastView) -> None:
    mapping = small_past_view.by_period[0]
    assert mapping.names == ("AAA", "BBB")
    assert mapping.to_series().to_list() == [1.0, 10.0]


def test_by_period_single_mapping_with_security_subset(
    small_past_view: PolarsPastView,
) -> None:
    subset = small_past_view.by_security[["BBB", "AAA"]]
    mapping = subset.by_period[0]
    assert mapping.names == ("BBB", "AAA")
    assert mapping.to_series().to_list() == [10.0, 1.0]


def test_by_period_slice(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_period[1:3]
    expected_periods = tuple(
        np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[us]")
    )
    assert subset.periods == expected_periods
    assert subset.by_security["AAA"].to_series().to_list() == [2.0, 3.0]
    assert subset.by_security["BBB"].to_series().to_list() == [20.0, 30.0]


def test_by_period_slice_non_contiguous(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_period[::2]
    expected_periods = tuple(
        np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[us]")
    )
    assert subset.periods == expected_periods
    assert subset.by_security["AAA"].to_series().to_list() == [1.0, 3.0]
    assert subset.by_security["BBB"].to_series().to_list() == [10.0, 30.0]


def test_before_after_inclusive(small_past_view: PolarsPastView) -> None:
    before = small_past_view.before("2024-01-02", inclusive=True)
    after = small_past_view.after("2024-01-02", inclusive=False)
    assert before.periods == tuple(
        np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[us]")
    )
    assert after.periods == tuple(np.array(["2024-01-03"], dtype="datetime64[us]"))


def test_between_closed_options(small_past_view: PolarsPastView) -> None:
    left = small_past_view.between("2024-01-01", "2024-01-03", closed="left")
    right = small_past_view.between("2024-01-01", "2024-01-03", closed="right")
    both = small_past_view.between("2024-01-01", "2024-01-03", closed="both")
    neither = small_past_view.between("2024-01-01", "2024-01-03", closed="neither")

    assert left.periods == tuple(
        np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[us]")
    )
    assert right.periods == tuple(
        np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[us]")
    )
    assert both.periods == tuple(
        np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[us]")
    )
    assert neither.periods == tuple(np.array(["2024-01-02"], dtype="datetime64[us]"))


def test_by_security_to_dataframe(small_past_view: PolarsPastView) -> None:
    df = small_past_view.by_security.to_dataframe(show_periods=True)
    assert df.columns == ["date", "AAA", "BBB"]


def test_by_security_to_dataframe_pandas(small_past_view: PolarsPastView) -> None:
    df = small_past_view.by_security.to_dataframe(backend="pandas")
    assert isinstance(df, pd.DataFrame)


def test_by_security_to_dataframe_invalid_backend(
    small_past_view: PolarsPastView,
) -> None:
    with pytest.raises(ValueError):
        small_past_view.by_security.to_dataframe(backend="invalid")  # type: ignore[arg-type]


def test_by_period_as_df_show_securities(small_past_view: PolarsPastView) -> None:
    df = small_past_view.by_period.as_df(show_securities=True, lazy=False)
    assert df.columns[0] == "security"
    assert df.select("security").to_series().to_list() == ["AAA", "BBB"]


def test_by_period_as_df_with_security_subset(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_security[["BBB", "AAA"]]
    df = subset.by_period.as_df(show_securities=True, lazy=False)
    assert df.columns[0] == "security"
    assert df.select("security").to_series().to_list() == ["BBB", "AAA"]


def test_by_period_to_dataframe_pandas(small_past_view: PolarsPastView) -> None:
    df = small_past_view.by_period.to_dataframe(backend="pandas")
    assert isinstance(df, pd.DataFrame)


def test_by_period_to_dataframe_invalid_backend(
    small_past_view: PolarsPastView,
) -> None:
    with pytest.raises(ValueError):
        small_past_view.by_period.to_dataframe(backend="invalid")  # type: ignore[arg-type]


def test_by_security_as_df_with_selection(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_security[["BBB", "AAA"]]
    df = subset.by_security.as_df(show_periods=False)
    assert df.columns == ["BBB", "AAA"]


def test_by_security_as_df_with_period_slice(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_period[1:3]
    df = subset.by_security.as_df(show_periods=True, lazy=False)
    assert df.columns == ["date", "AAA", "BBB"]
    assert df.select("AAA").to_series().to_list() == [2.0, 3.0]


def test_by_security_iteration_with_selection(small_past_view: PolarsPastView) -> None:
    subset = small_past_view.by_security[["BBB", "AAA"]]
    assert list(subset.by_security) == ["BBB", "AAA"]


def test_from_security_mappings_invalid_type() -> None:
    periods = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    ms = [
        {"AAA": decimal.Decimal("1.0")},
        {"AAA": decimal.Decimal("2.0")},
    ]
    with pytest.raises(ValueError):
        PolarsPastView.from_security_mappings(ms, periods)
