import backtest_lib as btl
from backtest_lib import target_weights
from backtest_lib.engine.decision import (
    Decision,
)
from backtest_lib.portfolio import uniform_portfolio


def test_over_single_market_movement(simple_market):
    market = simple_market
    initial_capital = 1_000_000
    initial_portfolio = uniform_portfolio(market.securities, value=initial_capital)

    def strategy(*args, **kwargs) -> Decision:
        return target_weights({"sec1": 0.5, "sec2": 0.5})

    backtest = btl.Backtest(
        strategy=strategy, market_view=market, initial_portfolio=initial_portfolio
    )
    results = backtest.run()

    nav = results.nav

    assert list(results.quantities_held.by_security["sec1"]) == [5000, 3750]
    assert list(results.quantities_held.by_security["sec2"]) == [50000, 75000]
    assert list(results.values_held.by_security["sec1"]) == [0.5 * nav[0], 0.5 * nav[1]]
    assert list(results.values_held.by_security["sec2"]) == [0.5 * nav[0], 0.5 * nav[1]]
    assert results.nav == [initial_capital, initial_capital * 1.5]
    assert results.total_return == 0.5
