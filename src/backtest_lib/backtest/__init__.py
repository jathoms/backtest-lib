from dataclasses import dataclass
import numpy as np

from backtest_lib.strategy import (
    MarketView,
    Strategy,
    WeightedPortfolio,
)
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Universe


@dataclass
class BacktestResults:
    total_growth: float


@dataclass
class BacktestSettings:
    allow_short: bool


class Backtest:
    strategy: Strategy
    universe: Universe
    market_view: MarketView
    initial_portfolio: WeightedPortfolio
    _current_portfolio: WeightedPortfolio
    settings: BacktestSettings | None

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: BacktestSettings | None = None,
    ):
        self.strategy = strategy
        self.universe = universe
        self.market_view = market_view
        self.initial_portfolio = initial_portfolio
        self.settings = settings

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults | None:
        results = BacktestResults(total_growth=1)
        self._current_portfolio = self.initial_portfolio

        past_market_view = self.market_view.truncated_to(1)
        today_prices = past_market_view.prices.close.by_period[-1]
        decision = self.strategy(
            universe=self.universe,
            current_portfolio=self._current_portfolio,
            market=past_market_view,
            ctx=ctx,
        )
        # assume we can perfectly track the target portfolio for now
        self._current_portfolio = decision.target
        for i in range(2, len(self.market_view.periods)):
            yesterday_prices = today_prices
            past_market_view = self.market_view.truncated_to(i)
            today_prices = past_market_view.prices.close.by_period[-1]
            pct_change = today_prices / yesterday_prices

            new_weights_vec = self._current_portfolio.holdings * pct_change
            new_total_weight = new_weights_vec.sum()
            results.total_growth *= new_total_weight
            new_weights_normed = new_weights_vec / new_total_weight
            new_cash_weight = 1 - new_weights_normed.sum()

            self._current_portfolio = WeightedPortfolio(
                cash=new_cash_weight,
                holdings=new_weights_normed,
            )

            decision = self.strategy(
                universe=self.universe,
                current_portfolio=self._current_portfolio,
                market=past_market_view,
                ctx=ctx,
            )
            target_portfolio = decision.target
            if not isinstance(target_portfolio, WeightedPortfolio):
                target_portfolio = target_portfolio.into_weighted()
            total_weight_after_decision = (
                decision.target.holdings.sum() + decision.target.cash
            )

            if not self.settings.allow_short and any(x > 0 for x in target_portfolio.holdings.values()):
                target_portfolio = target_portfolio.into_long_only()

            assert np.isclose(total_weight_after_decision, 1.0), (
                f"Total weight after making a decision cannot exceed 1.0, weight on period {i} was {total_weight_after_decision}"
            )

            # assume we can perfectly track the target portfolio for now
            self._current_portfolio = target_portfolio

        return results
