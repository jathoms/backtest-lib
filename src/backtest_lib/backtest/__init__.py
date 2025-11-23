from __future__ import annotations

import datetime as dt
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from backtest_lib.backtest.results import BacktestResults
from backtest_lib.backtest.settings import BacktestSettings
from backtest_lib.market import _BACKEND_PASTVIEW_MAPPING
from backtest_lib.strategy import Decision, MarketView, Strategy, WeightedPortfolio
from backtest_lib.strategy.context import StrategyContext

if TYPE_CHECKING:
    from backtest_lib.market import PastView
    from backtest_lib.universe import Universe
    from backtest_lib.universe.universe_mapping import UniverseMapping
    from backtest_lib.universe.vector_mapping import VectorMapping

logger = logging.getLogger(__name__)


def _to_pydt(some_datetime: Any) -> dt.datetime:
    if isinstance(some_datetime, dt.datetime):
        return some_datetime
    elif isinstance(some_datetime, np.datetime64):
        if np.isnat(some_datetime):
            raise ValueError("Cannot convert NaT to Python datetime.")
        us = some_datetime.astype("datetime64[us]").astype(np.int64)
        return dt.datetime.fromtimestamp(us / 1e6, dt.timezone.utc)
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
    _backend: type[PastView]

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: BacktestSettings = BacktestSettings.default(),
        *,
        backend="polars",
    ):
        self.strategy = strategy
        self.universe = universe
        self.market_view = market_view
        self.initial_portfolio = initial_portfolio
        self.settings = settings
        if backend not in _BACKEND_PASTVIEW_MAPPING:
            raise ValueError(f"Backtest backend '{backend}' not found.")
        self._backend = _BACKEND_PASTVIEW_MAPPING[backend]

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults:
        current_uni = None
        if ctx is None:
            ctx = StrategyContext()
        output_weights: list[VectorMapping[str, float]] = []
        returns_contribution: list[VectorMapping[str, float]] = []

        total_growth = 1.0
        self._current_portfolio = self.initial_portfolio
        yesterday_prices = self.market_view.prices.close.by_period[0]

        for i in range(1, len(self.market_view.periods) + 1):
            past_market_view = self.market_view.truncated_to(i)
            ctx.now = _to_pydt(self.market_view.periods[i - 1])
            logger.debug(
                f"Starting period {i} ({ctx.now}). Current total growth: {total_growth}"
            )
            decision = self.strategy(
                universe=self.universe,
                current_portfolio=self._current_portfolio,
                market=past_market_view,
                ctx=ctx,
            )
            if len(decision.target.holdings) < len(self.universe):
                # pad out the unnaccounted-for securities with 0.
                # NOTE: this is some extra allocation we might not need
                # as * 0 allocates, and so does + in the case where
                # keys are not equal.
                # maybe a .merge() method on the VectorMapping would make
                # more sense.
                logger.debug(
                    f"Incomplete universe returned by strategy (decision had {len(decision.target.holdings)}, "
                    f"full universe has {len(self.universe)}), filling remaining securities with 0."
                )
                object.__setattr__(
                    decision.target,
                    "holdings",
                    (
                        (self.market_view.prices.close.by_period[0] * 0)
                        + decision.target.holdings
                    ),
                )

            if current_uni is not None and set(decision.target.holdings.keys()) ^ set(
                current_uni
            ):
                print(
                    f"{ctx.now}: Universe changed! len: {len(current_uni)}->{len(decision.target.holdings.keys())}, diff: {set(current_uni) ^ set(decision.target.holdings.keys())}"
                )

            output_weights.append(decision.target.holdings)

            if past_market_view.tradable is not None:
                _check_tradable(
                    decision, past_market_view.tradable.by_period[-1], ctx.now
                )

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
            pct_change = today_prices / yesterday_prices
            returns_contribution.append(pct_change * target_portfolio.holdings)

            inter_day_adjusted_portfolio, growth = _apply_inter_period_price_changes(
                portfolio_after_decision, pct_change
            )

            total_growth *= growth

            self._current_portfolio = inter_day_adjusted_portfolio
            yesterday_prices = today_prices
            current_uni = list(decision.target.holdings.keys())

        allocation_history = self._backend.from_security_mappings(
            output_weights, self.market_view.periods
        )
        results = BacktestResults.from_weights_market_initial_capital(
            weights=allocation_history,
            market=self.market_view,
        )
        return results


def _apply_inter_period_price_changes(
    portfolio: WeightedPortfolio, pct_change: UniverseMapping[float]
) -> tuple[WeightedPortfolio, float]:
    prev_cash = portfolio.cash
    prev_hold = portfolio.holdings
    # logger.debug(
    #     f"Holdings length: {len(prev_hold)}, pct_change length: {len(pct_change)}, "
    #     f"hold: {prev_hold}, pct_change: {pct_change}"
    # )

    new_total_holdings_weight = prev_hold * pct_change
    new_total_weight = prev_cash + new_total_holdings_weight.sum()

    new_cash = prev_cash / new_total_weight
    new_holdings = new_total_holdings_weight / new_total_weight

    return WeightedPortfolio(
        cash=new_cash,
        holdings=new_holdings,
    ), new_total_weight


def _check_tradable(
    decision: Decision, tradable_mapping: UniverseMapping, now: dt.datetime
):
    tradable = {
        security for security, is_tradable in tradable_mapping.items() if is_tradable
    }
    logger.debug(f"Tradable len: {len(tradable)}")
    logger.debug(f"Decision weights len: {len(decision.target.holdings.keys())}")
    msgs = [
        f"Security '{sec}' is marked as non-tradable on period {now} but is given a value of {val}."
        for sec, val in decision.target.holdings.items()
        if val > 0 and sec not in tradable
    ]
    if msgs:
        warnings.warn("\n".join(msgs))
