import polars as pl
import pytest

from backtest_lib.market import MarketView


@pytest.fixture()
def prices_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [10.0, 11.0],
            "BBB": [20.0, 21.0],
        }
    )


@pytest.fixture()
def volume_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAA": [100, 110, 120],
            "BBB": [200, 210, 220],
        }
    )


def test_security_policy_unknown_raises(prices_df: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        MarketView(prices=prices_df, security_policy=object())  # type: ignore[arg-type]


def test_period_policy_unknown_raises(prices_df: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        MarketView(prices=prices_df, period_policy=object())  # type: ignore[arg-type]


def test_period_policy_strict_mismatch_raises(
    prices_df: pl.DataFrame, volume_df: pl.DataFrame
) -> None:
    with pytest.raises(ValueError):
        MarketView(prices=prices_df, volume=volume_df)
