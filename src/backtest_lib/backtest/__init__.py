from dataclasses import dataclass
import polars as pl

from backtest_lib.strategy import (
    MarketView,
    Strategy,
    StrategyContext,
    WeightedPortfolio,
)
from backtest_lib.universe import Universe

from backtest_lib.market.polars_impl import SeriesUniverseMapping


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
        total_value = 1
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
            # we're cheating here and using the specialised as_series method given by the polars impl
            # this will be generalised soon, maybe by adding vector math ops to the market structres
            pct_change: pl.Series = (
                today_prices.as_series() / yesterday_prices.as_series()
            )

            new_weights_vec = self._current_portfolio.weights.as_series() * pct_change
            new_total_weight = new_weights_vec.sum()
            total_value *= new_total_weight
            new_weights_normed = new_weights_vec / new_total_weight
            new_weights = SeriesUniverseMapping.from_names_and_data(
                self.universe, new_weights_normed
            )
            new_cash_weight = 1 - new_weights_normed.sum()

            self._current_portfolio = WeightedPortfolio(
                cash=new_cash_weight, weights=new_weights
            )

            decision = self.strategy(
                universe=self.universe,
                current_portfolio=self._current_portfolio,
                market=past_market_view,
                ctx=ctx,
            )
            # assume we can perfectly track the target portfolio for now
            self._current_portfolio = decision.target

        # again more cheating with polars series
        return BacktestResults(total_return=total_value)
