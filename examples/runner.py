from __future__ import annotations

from pathlib import Path

import polars as pl

import backtest_lib as btl
from backtest_lib.backtest.results import BacktestResults
from backtest_lib.strategy import Strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_CSV = REPO_ROOT / "docs/assets/data/spot_prices.csv"


def run_strategy(
    strategy: Strategy,
    market_csv: str | Path = DEFAULT_MARKET_CSV,
    *,
    initial_capital: float = 1_000_000,
) -> BacktestResults:
    prices = pl.read_csv(Path(market_csv))
    market = btl.MarketView(prices)
    initial_portfolio = btl.uniform_portfolio(market.securities, value=initial_capital)
    return btl.Backtest(
        strategy=strategy,
        market_view=market,
        initial_portfolio=initial_portfolio,
    ).run()
