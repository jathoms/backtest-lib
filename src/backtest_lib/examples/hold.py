"""A minimal strategy example that always holds."""

from backtest_lib import hold
from backtest_lib.engine.decision import Decision


def strategy() -> Decision:
    return hold()
