import datetime as dt

from backtest_lib.portfolio import WeightedPortfolio


class StrategyContext:
    __slots__ = ("now", "target_portfolio")
    target_portfolio: WeightedPortfolio
    now: dt.datetime | None
