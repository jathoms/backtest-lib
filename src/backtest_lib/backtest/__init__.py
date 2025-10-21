from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from backtest_lib.strategy import (
    MarketView,
    Strategy,
    WeightedPortfolio,
)
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Universe, UniverseMapping


@dataclass
class BacktestResults:
    total_growth: float


@dataclass
class BacktestSettings:
    allow_short: bool

    @staticmethod
    def default() -> BacktestSettings:
        return BacktestSettings(allow_short=False)


def _to_pydt(some_datetime: Any) -> dt.datetime:
    if isinstance(some_datetime, dt.datetime):
        return some_datetime
    elif isinstance(some_datetime, np.datetime64):
        if np.isnat(some_datetime):
            raise ValueError("Cannot convert NaT to Python datetime.")
        ns = some_datetime.astype("datetime64[ns]").astype(np.int64)
        return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc)
    else:
        raise TypeError(
            f"Cannot convert {some_datetime} with type {type(some_datetime)} to python datetime"
        )


class Backtest:
    strategy: Strategy
    universe: Universe
    market_view: MarketView
    initial_portfolio: WeightedPortfolio
    _current_portfolio: WeightedPortfolio
    settings: BacktestSettings

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: BacktestSettings = BacktestSettings.default(),
    ):
        self.strategy = strategy
        self.universe = universe
        self.market_view = market_view
        self.initial_portfolio = initial_portfolio
        self.settings = settings

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults:
        if ctx is None:
            ctx = StrategyContext()
        output_weights = []

        results = BacktestResults(total_growth=1)
        self._current_portfolio = self.initial_portfolio
        yesterday_prices = self.market_view.prices.close.by_period[0]

        for i in range(1, len(self.market_view.periods) + 1):
            past_market_view = self.market_view.truncated_to(i)
            ctx.now = _to_pydt(self.market_view.periods[i - 1])
            decision = self.strategy(
                universe=self.universe,
                current_portfolio=self._current_portfolio,
                market=past_market_view,
                ctx=ctx,
            )
            output_weights.append(decision.target.holdings)

            target_portfolio = decision.target
            if not isinstance(target_portfolio, WeightedPortfolio):
                target_portfolio = target_portfolio.into_weighted()
            total_weight_after_decision = (
                decision.target.holdings.sum() + decision.target.cash
            )

            assert np.isclose(total_weight_after_decision, 1.0), (
                f"Total weight after making a decision cannot exceed 1.0, weight on period {i} was {total_weight_after_decision}"
            )

            if not self.settings.allow_short and any(
                x < 0 for x in target_portfolio.holdings.values()
            ):
                target_portfolio = target_portfolio.into_long_only()

            # assume we can perfectly track the target portfolio for now
            portfolio_after_decision = target_portfolio
            today_prices = past_market_view.prices.close.by_period[-1]
            inter_day_adjusted_portfolio, growth = _apply_inter_period_price_changes(
                portfolio_after_decision, yesterday_prices, today_prices
            )

            results.total_growth *= growth

            self._current_portfolio = inter_day_adjusted_portfolio
            yesterday_prices = today_prices

        return results


def _apply_inter_period_price_changes(
    portfolio: WeightedPortfolio,
    prev_period_prices: UniverseMapping,
    new_period_prices: UniverseMapping,
) -> tuple[WeightedPortfolio, float]:
    pct_change = new_period_prices / prev_period_prices

    prev_cash = portfolio.cash
    prev_hold = portfolio.holdings

    new_total_holdings_weight = prev_hold * pct_change
    new_total_weight = prev_cash + new_total_holdings_weight.sum()

    new_cash = prev_cash / new_total_weight
    new_holdings = new_total_holdings_weight / new_total_weight

    return WeightedPortfolio(
        cash=new_cash,
        holdings=new_holdings,
    ), new_total_weight
