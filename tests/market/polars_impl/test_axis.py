import datetime

import numpy as np
import polars as pl
import pytest
from polars.exceptions import InvalidOperationError

from backtest_lib.market import Closed
from backtest_lib.market.polars_impl._axis import PeriodAxis, SecurityAxis


def test_static_constructor_security_axis():
    names = ["a", "b"]
    security_axis = SecurityAxis.from_names(names)
    assert security_axis.names == ("a", "b")
    assert security_axis.pos == {"a": 0, "b": 1}


def test_returning_length_of_names_security_axis():
    names = ["a", "b"]
    security_axis = SecurityAxis.from_names(names)
    assert len(security_axis.names) == 2


def test_ability_to_handle_empty_names_security_axis():
    security_axis = SecurityAxis.from_names([])
    assert len(security_axis.names) == 0


def test_casting_incompatible_data_to_date_should_throw_error_period_axis():
    s_string = pl.Series("string", ["a", "b", "c"])
    with pytest.raises(InvalidOperationError) as e_info:
        PeriodAxis.from_series(s_string)
    assert "conversion from `str` to `datetime[μs]` failed in column 'string'" in str(
        e_info.value
    )


def test_constructor_casting_required_period_axis():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    assert period_axis.labels == ("2023-01-01", "2023-01-02", "2023-01-03")
    assert period_axis.pos == {"2023-01-01": 0, "2023-01-02": 1, "2023-01-03": 2}
    expected_dt64_array = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[us]"
    )
    np.testing.assert_array_equal(period_axis.dt64, expected_dt64_array)


def test_len_period_axis():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    assert len(period_axis) == 3


def test_slicing_incontiguous_sequence():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    sliced_period_axis = period_axis.slice(slice(None, None, 2))
    assert sliced_period_axis.labels == ("2023-01-01", "2023-01-03")
    assert sliced_period_axis.pos == {"2023-01-01": 0, "2023-01-03": 1}
    expected_dt64_array = np.array(["2023-01-01", "2023-01-03"], dtype="datetime64[us]")
    np.testing.assert_array_equal(sliced_period_axis.dt64, expected_dt64_array)


def test_slicing_contiguous_sequence():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    sliced_period_axis = period_axis.slice(slice(1, 3))
    assert sliced_period_axis.labels == ("2023-01-02", "2023-01-03")
    assert sliced_period_axis.pos == {"2023-01-02": 0, "2023-01-03": 1}
    expected_dt64_array = np.array(["2023-01-02", "2023-01-03"], dtype="datetime64[us]")
    np.testing.assert_array_equal(sliced_period_axis.dt64, expected_dt64_array)


def test_bounds_after():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    assert period_axis.bounds_after(np.datetime64("2023-01-02"), inclusive=True) == (
        1,
        3,
    )
    assert period_axis.bounds_after(np.datetime64("2023-01-02"), inclusive=False) == (
        2,
        3,
    )


def test_bounds_before():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    assert period_axis.bounds_before(np.datetime64("2023-01-02"), inclusive=True) == (
        0,
        2,
    )
    assert period_axis.bounds_before(np.datetime64("2023-01-02"), inclusive=False) == (
        0,
        1,
    )


def test_bounds_between():
    series_date = pl.Series(
        "dates",
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ],
    )
    period_axis = PeriodAxis.from_series(series_date)
    assert period_axis.bounds_between(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-03"), closed=Closed.LEFT
    ) == (0, 2)
    assert period_axis.bounds_between(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-03"), closed="left"
    ) == (0, 2)
    assert period_axis.bounds_between(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-03"), closed=Closed.RIGHT
    ) == (1, 3)
    assert period_axis.bounds_between(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-03"), closed=Closed.BOTH
    ) == (0, 3)
    assert period_axis.bounds_between(
        np.datetime64("2023-01-02"), np.datetime64("2023-01-03"), closed=Closed.BOTH
    ) == (1, 3)
