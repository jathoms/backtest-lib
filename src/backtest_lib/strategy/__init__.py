from __future__ import annotations

from collections.abc import Callable

from backtest_lib.engine.decision import Decision

#: Callable strategy interface.
#: A strategy receives some combination of:
#: - Current universe,
#: - The portfolio state,
#: - A market view with time-fenced data, and
#: - A context object.
#: It returns a :data:`~backtest_lib.engine.decision.Decision` describing the target
#: allocation or action for the current decision point.
Strategy = Callable[..., Decision]
