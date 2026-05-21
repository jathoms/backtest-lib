import pytest

import backtest_lib as btl
from backtest_lib import target_weights
from backtest_lib.strategy import Decision


def test_no_lookahead_on_spike(spike_market):
    market = spike_market
    initial_capital = 1_000_000
    initial_portfolio = btl.uniform_portfolio(market.securities, value=initial_capital)

    # NB: adjust the signature to match your Strategy protocol — trim to just the
    # params the lib injects if it passes by name (your other test used zero args).
    def react_to_spike(universe, current_portfolio, market, ctx) -> Decision:
        sec1 = list(market.prices.close.by_security["sec1"])
        # Only act once we've observed two closes, and chase any up-move we SEE.
        if len(sec1) >= 2 and sec1[-1] > sec1[-2]:
            return target_weights({"sec1": 1.0})
        return target_weights({"sec2": 1.0})  # park in the flat asset (cash-like)

    backtest = btl.Backtest(
        strategy=react_to_spike,
        market_view=market,
        initial_portfolio=initial_portfolio,
    )
    results = backtest.run()

    # Look-ahead-free: the "chase" order, triggered by *observing* the jump on
    # 01-04, fills on the NEXT bar — by which point sec1 is already 200, so there's
    # nothing left to capture. NAV never moves; total return ~0.
    #
    # Leak (decision sees the spike bar, fills at the prior bar's 100): buys at 100,
    # rides to 200 -> ~+100%. That's the impossible profit this test exists to catch.
    assert results.total_return == pytest.approx(0.0, abs=1e-9), (
        f"look-ahead leak: profited from a spike only observable in hindsight "
        f"(total_return={results.total_return:.4f}; correct ~0, leak ~1.0)"
    )

    # Strongest, implementation-agnostic form: you can never get *ahead* of a move
    # you only saw after the fact.
    assert max(results.nav) == pytest.approx(initial_capital, abs=1.0)
