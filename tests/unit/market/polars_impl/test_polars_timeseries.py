import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtest_lib.market.polars_impl._axis import PeriodAxis
from backtest_lib.market.polars_impl._timeseries import AlignmentError, PolarsTimeseries


@pytest.fixture()
def small_timeseries() -> PolarsTimeseries:
    return PolarsTimeseries.from_vectors(
        values=[1.0, 2.0, 3.0],
        periods=[
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
        ],
        name="price",
    )


def test_from_vectors_success() -> None:
    ts = PolarsTimeseries.from_vectors(
        values=[1, 2, 3],
        periods=[
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
        ],
        name="qty",
    )
    assert ts.to_series().to_list() == [1, 2, 3]
    assert ts.to_series().dtype == pl.Int64


def test_from_vectors_length_mismatch() -> None:
    with pytest.raises(AlignmentError):
        PolarsTimeseries.from_vectors(
            values=[1.0, 2.0],
            periods=[np.datetime64("2024-01-01")],
            name="price",
        )


def test_from_vectors_unsupported_dtype() -> None:
    with pytest.raises(TypeError):
        PolarsTimeseries.from_vectors(  # type: ignore[arg-type]
            values=["a", "b"],
            periods=[np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        )


def test_constructor_misaligned_axis() -> None:
    vec = pl.Series("value", [1.0, 2.0])
    axis = PeriodAxis.from_series(
        pl.Series(
            np.asarray(
                [
                    np.datetime64("2024-01-01"),
                    np.datetime64("2024-01-02"),
                    np.datetime64("2024-01-03"),
                ],
                dtype="datetime64[us]",
            ),
            dtype=pl.Datetime("us"),
        )
    )
    with pytest.raises(ValueError):
        PolarsTimeseries(vec, axis, "price", float)


def test_getitem_scalar(small_timeseries: PolarsTimeseries) -> None:
    assert small_timeseries[0] == 1.0


def test_getitem_slice(small_timeseries: PolarsTimeseries) -> None:
    sliced = small_timeseries[1:3]
    assert sliced.to_series().to_list() == [2.0, 3.0]
    assert len(sliced) == 2


def test_before_after_between(small_timeseries: PolarsTimeseries) -> None:
    before = small_timeseries.before("2024-01-02", inclusive=False)
    after = small_timeseries.after("2024-01-02", inclusive=True)
    between = small_timeseries.between("2024-01-01", "2024-01-03", closed="both")

    assert before.to_series().to_list() == [1.0]
    assert after.to_series().to_list() == [2.0, 3.0]
    assert between.to_series().to_list() == [1.0, 2.0, 3.0]


def test_arithmetic_with_scalar(small_timeseries: PolarsTimeseries) -> None:
    added = small_timeseries + 1
    multiplied = 2 * small_timeseries
    divided = small_timeseries / 2
    assert added.to_series().to_list() == [2.0, 3.0, 4.0]
    assert multiplied.to_series().to_list() == [2.0, 4.0, 6.0]
    assert divided.to_series().to_list() == [0.5, 1.0, 1.5]


def test_arithmetic_with_timeseries(small_timeseries: PolarsTimeseries) -> None:
    other = PolarsTimeseries.from_vectors(
        values=[10.0, 20.0, 30.0],
        periods=[
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
        ],
        name="other",
    )
    combined = small_timeseries + other
    assert combined.to_series().to_list() == [11.0, 22.0, 33.0]


def test_axis_mismatch_raises() -> None:
    left = PolarsTimeseries.from_vectors(
        values=[1.0, 2.0],
        periods=[np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
    )
    right = PolarsTimeseries.from_vectors(
        values=[3.0, 4.0],
        periods=[np.datetime64("2024-01-02"), np.datetime64("2024-01-03")],
    )
    with pytest.raises(ValueError):
        _ = left + right


def test_sum_mean_abs_floor_truncate(small_timeseries: PolarsTimeseries) -> None:
    assert small_timeseries.sum() == 6.0
    assert small_timeseries.mean() == 2.0
    assert small_timeseries.abs().to_series().to_list() == [1.0, 2.0, 3.0]
    assert small_timeseries.floor().to_series().dtype == pl.Int64
    assert small_timeseries.truncate().to_series().dtype == pl.Int64


def test_to_series_backends(small_timeseries: PolarsTimeseries) -> None:
    polars_series = small_timeseries.to_series()
    pandas_series = small_timeseries.to_series(backend="pandas")
    assert isinstance(polars_series, pl.Series)
    assert isinstance(pandas_series, pd.Series)
    assert pandas_series.tolist() == [1.0, 2.0, 3.0]
