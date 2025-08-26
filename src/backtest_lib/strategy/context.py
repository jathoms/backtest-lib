from dataclasses import dataclass
from backtest_lib.strategy import WeightedPortfolio


@dataclass
class StrategyContext:
    target_portfolio: WeightedPortfolio
