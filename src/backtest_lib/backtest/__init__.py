from __future__ import annotations

import datetime as dt
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from backtest_lib.backtest._helpers import _to_pydt
from backtest_lib.backtest.results import BacktestResults
from backtest_lib.backtest.schedule import DecisionSchedule
from backtest_lib.backtest.schedule import decision_schedule as make_decision_schedule
from backtest_lib.backtest.settings import BacktestSettings
from backtest_lib.market import get_pastview_from_mapping
from backtest_lib.strategy import Decision, MarketView, Strategy, WeightedPortfolio
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
    :class:`PastView <backtest_lib.pastview.PastView>` backend and converted into
    :class:`BacktestResults <backtest_lib.backtest.results.BacktestResults>`.

    The simulation model is intentionally simple:

    - Decisions are evaluated on a
      :class:`DecisionSchedule <backtest_lib.schedule.DecisionSchedule>`. Between
      decision points, the portfolio weights drift only due to price movements.
    - The backtest assumes the portfolio can be rebalanced to the strategy's
      target portfolio exactly at decision times (i.e., no slippage/fees unless
      incorporated elsewhere).
    - Strategies may return an incomplete set of holdings; missing securities
      are padded with zero weight to match the full universe.
    - If :attr:`BacktestSettings.allow_short
      <backtest_lib.backtest.settings.BacktestSettings.allow_short>`
      is False and the target contains negative weights, the target is coerced into
      a long-only portfolio.

    Args:
        strategy: Callable strategy that produces a
            :class:`Decision <backtest_lib.decision.Decision>` given the current
            universe, portfolio, market view, and optional context.
        universe: The tradable security set defining the expected holdings keys.
            See :class:`Universe <backtest_lib.universe.Universe>`.
        market_view: Historical market data used for pricing and period iteration.
            See :class:`MarketView <backtest_lib.market.MarketView>`. The decision
            schedule defaults to ``market_view.periods`` when not provided.
        initial_portfolio: Starting
            :class:`WeightedPortfolio <backtest_lib.portfolio.WeightedPortfolio>`
            used to initialize the simulation state.
        settings: Controls simulation constraints (e.g., shorting). Defaults to
            :meth:`BacktestSettings.default
            <backtest_lib.backtest.settings.BacktestSettings.default>`.
        decision_schedule: Rebalance schedule. May be:

            - A :class:`DecisionSchedule
              <backtest_lib.schedule.DecisionSchedule>` instance,
            - A string specification consumed by
              :func:`make_decision_schedule
              <backtest_lib.schedule.make_decision_schedule>`, or
            - None, in which case a schedule is constructed from
              ``market_view.periods``.
        backend: Backend identifier used to select the
        :class:`PastView <backtest_lib.pastview.PastView>`
            implementation used for data manipulation, memory allocation,
            and results view.
            Default (and currently only implemented) backend is "polars"

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
        >>> results.annualized_return  # doctest: +ELLIPSIS
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

    def __init__(
        self,
        strategy: Strategy,
        universe: Universe,
        market_view: MarketView,
        initial_portfolio: WeightedPortfolio,
        settings: BacktestSettings = _DEFAULT_BACKTEST_SETTINGS,
        *,
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

    def run(self, ctx: StrategyContext | None = None) -> BacktestResults:
        schedule_it = iter(self._schedule)
        next_decision_period = next(schedule_it)
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
                f"Starting period {i} ({ctx.now}). Current total growth:"
                f" {total_growth}",
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
                if len(decision.target.holdings) < len(self.universe):
                    # pad out the unnaccounted-for securities with 0.
                    # NOTE: this is some extra allocation we might not need
                    # as * 0 allocates, and so does + in the case where
                    # keys are not equal.
                    # maybe a .merge() method on the VectorMapping would make
                    # more sense.
                    logger.debug(
                        "Incomplete universe returned by strategy (decision had"
                        f" {len(decision.target.holdings)}, full universe has"
                        f" {len(self.universe)}), filling remaining securities with 0.",
                    )
                    object.__setattr__(
                        decision.target,
                        "holdings",
                        (
                            (self.market_view.prices.close.by_period[0] * 0)
                            + decision.target.holdings
                        ),
                    )

                if current_uni is not None and set(
                    decision.target.holdings.keys(),
                ) ^ set(current_uni):
                    logger.debug(
                        f"{ctx.now}: Universe changed! len:"
                        f" {len(current_uni)}->{len(decision.target.holdings.keys())},"
                        " diff:"
                        f" {set(current_uni) ^ set(decision.target.holdings.keys())}",
                    )

                if past_market_view.tradable is not None:
                    _check_tradable(
                        decision,
                        past_market_view.tradable.by_period[-1],
                        ctx.now,
                    )

                target_portfolio = decision.target
                if not isinstance(target_portfolio, WeightedPortfolio):
                    target_portfolio = target_portfolio.into_weighted()
                total_weight_after_decision = (
                    decision.target.holdings.sum() + decision.target.cash
                )

                assert np.isclose(total_weight_after_decision, 1.0), (
                    "Total weight after making a decision cannot exceed 1.0, "
                    f"weight on period {i} was {total_weight_after_decision}"
                )

                if not self.settings.allow_short and any(
                    x < 0 for x in target_portfolio.holdings.values()
                ):
                    target_portfolio = target_portfolio.into_long_only()

                # assume we can perfectly track the target portfolio for now
                portfolio_after_decision = target_portfolio
            else:
                portfolio_after_decision = self._current_portfolio
            output_weights.append(portfolio_after_decision.holdings)

            today_prices = past_market_view.prices.close.by_period[-1]
            pct_change = today_prices / yesterday_prices
            returns_contribution.append(pct_change * portfolio_after_decision.holdings)

            inter_day_adjusted_portfolio, growth = _apply_inter_period_price_changes(
                portfolio_after_decision,
                pct_change,
            )

            total_growth *= growth

            self._current_portfolio = inter_day_adjusted_portfolio
            yesterday_prices = today_prices
            current_uni = list(portfolio_after_decision.holdings.keys())

        allocation_history = self._backend.from_security_mappings(
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

    return (
        WeightedPortfolio(
            cash=new_cash,
            holdings=new_holdings,
        ),
        new_total_weight,
    )


def _check_tradable(
    decision: Decision,
    tradable_mapping: UniverseMapping,
    now: dt.datetime,
):
    tradable = {
        security for security, is_tradable in tradable_mapping.items() if is_tradable
    }
    logger.debug(f"Tradable len: {len(tradable)}")
    logger.debug(f"Decision weights len: {len(decision.target.holdings.keys())}")
    msgs = [
        f"Security '{sec}' is marked as non-tradable on period {now} but is given a"
        f" value of {val}."
        for sec, val in decision.target.holdings.items()
        if val > 0 and sec not in tradable
    ]
    if msgs:
        warnings.warn("\n".join(msgs), stacklevel=2)
