import pytest

import backtest_lib as btl
from backtest_lib.engine.decision import (
    Decision,
    hold,
)
from backtest_lib.portfolio import WeightedPortfolio


def test_returns_track_price_when_portfolio_is_one_security(single_security_market):
    market = single_security_market
    security = single_security_market.securities[0]
    initial_capital = 1_000_000
    initial_portfolio = WeightedPortfolio(
        universe=market.securities,
        holdings={security: 1.0},
        cash=0,
        total_value=initial_capital,
    )

    def strategy(*args, **kwargs) -> Decision:
        return hold()

    backtest = btl.Backtest(
        strategy=strategy, market_view=market, initial_portfolio=initial_portfolio
    )
    results = backtest.run()
    assert all(
        [
            portfolio_value / initial_capital == pytest.approx(i)
            for portfolio_value, i in zip(results.nav, range(1, 101), strict=True)
        ]
    )

    assert results.total_return == pytest.approx(99)
