from dataclasses import dataclass
from backtest_lib.portfolio import WeightedPortfolio


@dataclass
class StrategyContext:
    target_portfolio: WeightedPortfolio
