from pathlib import Path

import pytest
from polars import read_csv

from backtest_lib import MarketView
from backtest_lib.market._backends import (
    _get_mapping_type_from_backend,
    _get_pastview_type_from_backend,
    _get_timeseries_type_from_backend,
)

MARKET_BACKENDS = ["polars", "native"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmarks (opt-in)",
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def single_security_market(test_data_dir) -> MarketView:
    data = read_csv(test_data_dir / "single_security.csv")
    market = MarketView(data)
    return market


@pytest.fixture(scope="session")
def simple_market(test_data_dir) -> MarketView:
    data = read_csv(test_data_dir / "simple_market.csv")
    market = MarketView(data)
    return market


@pytest.fixture(scope="session")
def spike_market(test_data_dir) -> MarketView:
    data = read_csv(test_data_dir / "spike_market.csv")
    return MarketView(data)


@pytest.fixture(params=MARKET_BACKENDS)
def market_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture()
def mapping_type(market_backend: str):
    return _get_mapping_type_from_backend(market_backend)


@pytest.fixture()
def past_view_type(market_backend: str):
    return _get_pastview_type_from_backend(market_backend)


@pytest.fixture()
def timeseries_type(market_backend: str):
    return _get_timeseries_type_from_backend(market_backend)
