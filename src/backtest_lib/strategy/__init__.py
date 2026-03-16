from __future__ import annotations

from collections.abc import Callable

from backtest_lib.engine.decision import Decision

#: Callable strategy interface.
#: A strategy receives some combination of:
#:
#: - Current universe,
#:
#: - The portfolio state,
#:
#: - A market view with time-fenced data, and
#:
#: - A context object.
#:
#: Arguments are injected by parameter name (pytest-fixture style). A strategy may
#: request any subset of ``universe``, ``current_portfolio``, ``market``, and
#: ``ctx``.
#:
#: Injected parameter types:
#:
#: - ``universe``: ``tuple[str, ...]``
#: - ``current_portfolio``: :class:`~backtest_lib.portfolio.Portfolio`
#: - ``market``: :class:`~backtest_lib.market.MarketView`
#: - ``ctx``: :class:`~backtest_lib.strategy.context.StrategyContext`
#:
#: Examples:
#:
#: .. code-block:: python
#:
#:    def example_strategy(universe):
#:        return do_something_with(universe)
#:
#:    def rebalance(current_portfolio, market):
#:        return target_weights(
#:            some_portfolio_aware_computation(
#:                current_portfolio.holdings,
#:                market.prices.close,
#:            )
#:        )
#:
#:    def with_context(universe, ctx):
#:        security_to_buy = day_specific_computation(ctx.now)
#:        return trade("BUY", 1, security_to_buy)
#:
#: Return type:
#:
#: - :data:`~backtest_lib.engine.decision.Decision`
#:   (a union of decision objects such as hold, trade, target weights/holdings,
#:   and reallocate decisions)
#:
#: The functions that create
#: :data:`~backtest_lib.engine.decision.Decision` objects are documented in
#: :ref:`Decision <decision-reference>`.
Strategy = Callable[..., Decision]
