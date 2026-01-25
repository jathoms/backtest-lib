import polars as pl
import pytest

from backtest_lib.market import MarketView
from backtest_lib.market.polars_impl import PolarsPastView


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
def tradable_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [1, 0],
            "BBB": [1, 1],
        }
    )


@pytest.fixture()
def volume_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [100, 110],
            "BBB": [200, 210],
        }
    )


@pytest.fixture()
def signal_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAA": [0.1, 0.2],
            "BBB": [0.3, 0.4],
        }
    )


def test_prices_from_past_view(prices_df: pl.DataFrame) -> None:
    past = PolarsPastView.from_dataframe(prices_df)
    view = MarketView(prices=past)
    assert view.prices.close is past


def test_tradable_volume_from_dataframe(
    prices_df: pl.DataFrame, tradable_df: pl.DataFrame, volume_df: pl.DataFrame
) -> None:
    view = MarketView(prices=prices_df, tradable=tradable_df, volume=volume_df)
    assert view.tradable is not None
    assert view.volume is not None
    assert view.tradable.securities == ("AAA", "BBB")
    assert view.volume.securities == ("AAA", "BBB")


def test_signals_are_aligned(prices_df: pl.DataFrame, signal_df: pl.DataFrame) -> None:
    view = MarketView(prices=prices_df, signals={"alpha": signal_df})
    assert "alpha" in view.signals
    assert view.signals["alpha"].securities == view.securities
    assert view.signals["alpha"].periods == view.periods


def test_resolve_axis_spec_variants(
    prices_df: pl.DataFrame, tradable_df: pl.DataFrame, volume_df: pl.DataFrame
) -> None:
    view = MarketView(
        prices=prices_df,
        tradable=tradable_df,
        volume=volume_df,
        signals={"alpha": prices_df},
    )
    assert view._resolve_axis_spec("tradable") is view.tradable
    assert view._resolve_axis_spec("volume") is view.volume
    assert view._resolve_axis_spec("signal:alpha") is view.signals["alpha"]


def test_resolve_axis_spec_unknown(prices_df: pl.DataFrame) -> None:
    view = MarketView(prices=prices_df)
    with pytest.raises(ValueError):
        view._resolve_axis_spec("unknown")


def test_resolve_axis_spec_none(prices_df: pl.DataFrame) -> None:
    view = MarketView(prices=prices_df)
    with pytest.raises(ValueError):
        view._resolve_axis_spec("tradable")


def test_truncated_to(
    prices_df: pl.DataFrame, tradable_df: pl.DataFrame, volume_df: pl.DataFrame
) -> None:
    view = MarketView(
        prices=prices_df,
        tradable=tradable_df,
        volume=volume_df,
        signals={"alpha": prices_df},
    )
    truncated = view.truncated_to(1)
    assert len(truncated.periods) == 1
    assert truncated.tradable is not None
    assert truncated.volume is not None
    assert len(truncated.tradable.periods) == 1
    assert len(truncated.volume.periods) == 1
    assert len(truncated.signals["alpha"].periods) == 1


def test_filter_securities_missing_optional(prices_df: pl.DataFrame) -> None:
    view = MarketView(prices=prices_df, signals={"alpha": prices_df})
    filtered = view.filter_securities(["BBB"])
    assert filtered.prices.close.securities == ("BBB",)
    assert filtered.signals["alpha"].securities == ("BBB",)
