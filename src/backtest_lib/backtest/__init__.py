from __future__ import annotations

import datetime as dt
import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from backtest_lib.market import PastView, PeriodIndex
from backtest_lib.market.polars_impl import PolarsPastView
from backtest_lib.market.timeseries import Timeseries
from backtest_lib.strategy import Decision, MarketView, Strategy, WeightedPortfolio
from backtest_lib.strategy.context import StrategyContext
from backtest_lib.universe import Universe, UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.universe.vector_mapping import VectorMapping

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    total_growth: float
    allocation_history: PastView[UniverseMapping[float], Timeseries, PeriodIndex]
    pnl_history: PastView[UniverseMapping[float], Timeseries, PeriodIndex]


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


_BACKEND_PASTVIEW_MAPPING: dict[str, type[PastView[Any, Any, Any]]] = {
    "polars": PolarsPastView
}


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
        pnl_history: list[VectorMapping[str, float]] = []

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
            if len(decision.target.holdings) > len(self.universe):
                print(
                    f"{ctx.now}: hold:{len(decision.target.holdings)}, univ:{len(self.universe)}"
                )
            if len(decision.target.holdings) < len(self.universe):
                # pad out the unnaccounted-for securities with 0.
                # NOTE: this is some extra allocation we might not need
                # as zeroed allocates, and so does + in the case where
                # keys are not equal.
                # maybe a .merge() method on the VectorMapping would make
                # more sense.
                object.__setattr__(
                    decision.target,
                    "holdings",
                    (
                        self.market_view.prices.close.by_period[0].zeroed()
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
            pnl_history.append(pct_change * target_portfolio.holdings)

            inter_day_adjusted_portfolio, growth = _apply_inter_period_price_changes(
                portfolio_after_decision, pct_change
            )

            total_growth *= growth

            self._current_portfolio = inter_day_adjusted_portfolio
            yesterday_prices = today_prices
            current_uni = list(decision.target.holdings.keys())

        results = BacktestResults(
            total_growth=total_growth,
            allocation_history=self._backend.from_security_mappings(
                output_weights, self.market_view.periods
            ),
            pnl_history=self._backend.from_security_mappings(
                pnl_history, self.market_view.periods
            ),
        )
        return results


def _apply_inter_period_price_changes(
    portfolio: WeightedPortfolio, pct_change: UniverseMapping
) -> tuple[WeightedPortfolio, float]:
    prev_cash = portfolio.cash
    prev_hold = portfolio.holdings
    logger.debug(f"Holdings length: {len(prev_hold)}")

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
