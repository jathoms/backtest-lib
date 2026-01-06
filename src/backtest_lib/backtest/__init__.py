from __future__ import annotations

import datetime as dt
import logging
import warnings
from typing import TYPE_CHECKING

from backtest_lib.backtest._helpers import _to_pydt
from backtest_lib.backtest.results import BacktestResults
from backtest_lib.backtest.schedule import DecisionSchedule, make_decision_schedule
from backtest_lib.backtest.settings import BacktestSettings
from backtest_lib.engine import Engine, make_engine
from backtest_lib.engine.execute.perfect_world import PerfectWorldPlanExecutor
from backtest_lib.engine.plan.perfect_world import (
    PerfectWorldPlanGenerator,
)
from backtest_lib.market import MarketView, get_pastview_from_mapping
from backtest_lib.portfolio import Portfolio, WeightedPortfolio
from backtest_lib.strategy import Strategy
from backtest_lib.strategy.context import StrategyContext

if TYPE_CHECKING:
    from backtest_lib.market import PastView
    from backtest_lib.universe import Universe
    from backtest_lib.universe.universe_mapping import UniverseMapping
    from backtest_lib.universe.vector_mapping import VectorMapping

logger = logging.getLogger(__name__)

_DEFAULT_BACKTEST_SETTINGS = BacktestSettings.default()


class Backtest:
    """Runs a historical simulation of a :class:`Strategy
    <backtest_lib.strategy.Strategy>`
    over a :class:`MarketView <backtest_lib.market.MarketView>` and a
    :class:`Universe <backtest_lib.universe.Universe>`.

    :class:`Backtest <backtest_lib.backtest.Backtest>` is a lightweight orchestration
    object that iterates through market periods, invokes the strategy on configured
    decision dates, and updates the simulated portfolio by applying inter-period
    price changes. The resulting allocation history is materialized via a configurable
    :class:`PastView <backtest_lib.market.PastView>` backend and converted into
    :class:`BacktestResults <backtest_lib.backtest.results.BacktestResults>`.

    The simulation model is intentionally simple:

    - Decisions are evaluated on a
      :class:`DecisionSchedule <backtest_lib.backtest.schedule.DecisionSchedule>`.
      Between decision points, the portfolio weights drift only due to price movements.
    - The backtest assumes the portfolio can be rebalanced to the strategy's
      target portfolio exactly at decision times (i.e., no slippage/fees unless
      incorporated elsewhere).
    - Strategies may return an incomplete set of holdings; missing securities
      are padded with zero weight to match the full universe.
    - If ``BacktestSettings.allow_short``
      is False and the target contains negative weights, the target is coerced into
      a long-only portfolio.

    Args:
        strategy: Callable strategy that produces a
            :class:`~backtest_lib.strategy.Decision` given the current
            universe, portfolio, market view, and optional context.
        universe: The tradable security set defining the expected holdings keys.
            See :class:`~backtest_lib.universe.Universe`.
        market_view: Historical market data used for pricing and period iteration.
            See :class:`~backtest_lib.market.MarketView`. The decision
            schedule defaults to ``market_view.periods`` when not provided.
        initial_portfolio: Starting
            :class:`~backtest_lib.portfolio.WeightedPortfolio`
            used to initialize the simulation state.
        settings: Controls simulation constraints (e.g., shorting). Defaults to
            :meth:`BacktestSettings.default
            <backtest_lib.backtest.settings.BacktestSettings.default>`.
        decision_schedule: Rebalance schedule. May be:

            - A :class:`~backtest_lib.backtest.schedule.DecisionSchedule` instance,
            - A string specification consumed by
              :func:`~backtest_lib.backtest.schedule.make_decision_schedule`, or
            - None, in which case a schedule is constructed from
              ``market_view.periods``.

        backend: Backend identifier used to select the
            :class:`~backtest_lib.market.PastView`
            implementation used for data manipulation, memory allocation,
            and results view.
            Default (and currently only implemented) backend is "polars".

    Attributes:
        strategy: Strategy under test.
        universe: Universe used for padding/consistency checks.
        market_view: Market data used during the run.
        initial_portfolio: Starting portfolio at the beginning of the run.
        settings: Backtest configuration settings.

    Example:
        >>> import backtest_lib as btl
        >>> from polars import read_csv
        >>> from backtest_lib.portfolio import uniform_portfolio
        >>> spot_prices = read_csv("docs/assets/data/spot_prices.csv")
        >>> market = btl.MarketView(spot_prices)
        >>> universe = market.securities
        >>> def hold_strategy(universe, current_portfolio, market, ctx):
        ...     return btl.Decision(current_portfolio)
        >>> bt = btl.Backtest(
        ...     hold_strategy, universe, market, uniform_portfolio(universe)
        ... )
        >>> results = bt.run()
        >>> results.annualized_return
        -0.00022650...
    """

    strategy: Strategy
    universe: Universe
    market_view: MarketView
    initial_portfolio: WeightedPortfolio
    _current_portfolio: WeightedPortfolio
    settings: BacktestSettings
    _schedule: DecisionSchedule
    _backend: type[PastView]
    _engine: Engine

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: BacktestSettings = _DEFAULT_BACKTEST_SETTINGS,
        *,
        engine: Engine | None = None,
        decision_schedule: str | DecisionSchedule | None = None,
        backend="polars",
    ):
        self.strategy = strategy
        self.universe = universe
        self.market_view = market_view
        self.initial_portfolio = initial_portfolio
        self.settings = settings
        if isinstance(decision_schedule, str):
            self._schedule = make_decision_schedule(
                decision_schedule,
                start=market_view.periods[0],
                end=market_view.periods[-1],
            )
        elif decision_schedule is None:
            self._schedule = make_decision_schedule(market_view.periods)
        else:
            self._schedule = decision_schedule
        self._backend = get_pastview_from_mapping(backend)

        self._engine = engine or make_engine(
            PerfectWorldPlanGenerator(backend, universe),
            PerfectWorldPlanExecutor(backend, universe),
        )

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults:
        schedule_it = iter(self._schedule)
        next_decision_period = next(schedule_it)
        if ctx is None:
            ctx = StrategyContext()
        output_weights: list[VectorMapping[str, float]] = []
        returns_contribution: list[VectorMapping[str, float]] = []

        self._current_portfolio = self.initial_portfolio
        yesterday_prices = self.market_view.prices.close.by_period[0]

        for i in range(1, len(self.market_view.periods) + 1):
            past_market_view = self.market_view.truncated_to(i)
            today_prices = past_market_view.prices.close.by_period[-1]
            ctx.now = _to_pydt(self.market_view.periods[i - 1])
            logger.debug(
                f"Starting period {i} ({ctx.now}). Current total growth:"
                f" {self._current_portfolio.total_value}",
            )
            if ctx.now >= _to_pydt(next_decision_period):
                try:
                    next_decision_period = next(schedule_it)
                except StopIteration:
                    logger.debug(
                        "Reached end of decision schedule, breaking from backtest loop"
                        f" at {ctx.now} (period {i}).",
                    )
                    break

                decision = self.strategy(
                    universe=self.universe,
                    current_portfolio=self._current_portfolio,
                    market=past_market_view,
                    ctx=ctx,
                )
                # NOTE: we are using close prices here. this is an implicit assumption.
                # the user may want to use (low+high)/2, mid price, VWAP/TWAP.
                decision_result = self._engine.execute_decision(
                    decision=decision,
                    portfolio=self._current_portfolio,
                    prices=today_prices,
                    market=self.market_view,
                )

                portfolio_after_decision = decision_result.after

                if past_market_view.tradable is not None:
                    _check_tradable(
                        portfolio_after_decision,
                        past_market_view.tradable.by_period[-1],
                        ctx.now,
                    )
            else:
                portfolio_after_decision = self._current_portfolio
            output_weights.append(portfolio_after_decision.holdings)

            pct_change = today_prices / yesterday_prices
            returns_contribution.append(pct_change * portfolio_after_decision.holdings)

            inter_day_adjusted_portfolio = _apply_inter_period_price_changes(
                portfolio_after_decision.into_weighted(),
                pct_change,
            )

            self._current_portfolio = inter_day_adjusted_portfolio
            yesterday_prices = today_prices

        allocation_history: PastView = self._backend.from_security_mappings(
            output_weights,
            self.market_view.periods[: i - 1],
        )
        results = BacktestResults.from_weights_market_initial_capital(
            weights=allocation_history,
            market=self.market_view.truncated_to(i - 1),
            backend=self._backend,
        )
        return results


def _apply_inter_period_price_changes(
    portfolio: WeightedPortfolio,
    pct_change: UniverseMapping[float],
) -> WeightedPortfolio:
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
        total_value=portfolio.total_value * new_total_weight,
    )


def _check_tradable(
    portfolio_after_decision: Portfolio,
    tradable_mapping: UniverseMapping,
    now: dt.datetime,
):
    tradable = {
        security for security, is_tradable in tradable_mapping.items() if is_tradable
    }
    logger.debug(f"Tradable len: {len(tradable)}")
    logger.debug(
        f"New portfolio weights len:  {len(portfolio_after_decision.holdings.keys())}"
    )
    msgs = [
        f"Security '{sec}' is marked as non-tradable on period {now} but is given a"
        f" value of {val}."
        for sec, val in portfolio_after_decision.holdings.items()
        if val > 0 and sec not in tradable
    ]
    if msgs:
        warnings.warn("\n".join(msgs), stacklevel=2)
