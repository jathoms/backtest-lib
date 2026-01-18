from pathlib import Path

import pytest
from polars import read_csv

from backtest_lib import MarketView


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def single_security_market(test_data_dir) -> MarketView:
    data = read_csv(test_data_dir / "single_security.csv")
    market = MarketView(data)
    return market
