from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from backtest_lib.engine.decision import Decision
    from backtest_lib.market import MarketView
    from backtest_lib.portfolio import Portfolio
    from backtest_lib.strategy.context import StrategyContext
    from backtest_lib.universe import Universe


@runtime_checkable
class Strategy(Protocol):
    """Callable strategy interface.

    A strategy receives the current universe, the portfolio state, a market view
    with time-fenced data, and an optional context object. It returns a
    :class:`~backtest_lib.engine.decision.Decision` describing the target
    allocation or action for the current decision point.
    """

    def __call__(
        self,
        universe: Universe,
        current_portfolio: Portfolio,
        market: MarketView,
        ctx: StrategyContext | None,
    ) -> Decision: ...
