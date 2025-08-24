from dataclasses import dataclass

from backtest_lib.strategy import (
    MarketView,
    Strategy,
    StrategyContext,
    WeightedPortfolio,
)
from backtest_lib.universe import Universe


@dataclass
class BacktestResults:
    total_return: float


class Backtest:
    strategy: Strategy
    universe: Universe
    market_view: MarketView
    initial_portfolio: WeightedPortfolio
    _current_portfolio: WeightedPortfolio
    settings: dict | None = None

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: dict | None = None,
    ):
        self.strategy = strategy
        self.universe = universe
        self.market_view = market_view
        self.initial_portfolio = initial_portfolio
        self.settings = settings

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults | None:
        total_value = 1.0
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
            total_value *= new_total_weight
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
            # assume we can perfectly track the target portfolio for now
            self._current_portfolio = decision.target

        return BacktestResults(total_return=total_value)
