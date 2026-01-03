from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from backtest_lib.market import MarketView
    from backtest_lib.portfolio import WeightedPortfolio
    from backtest_lib.strategy.context import StrategyContext
    from backtest_lib.strategy.decision import Decision
    from backtest_lib.universe import Universe


@runtime_checkable
class Strategy(Protocol):
    """PLACEHOLDER"""

    def __call__(
        self,
        universe: Universe,
        current_portfolio: WeightedPortfolio,
        market: MarketView,
        ctx: StrategyContext | None,
    ) -> Decision: ...
