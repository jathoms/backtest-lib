"""Importable example strategies.

This package is intentionally tiny so users can import simple reference
strategies without pulling in notebook/data-heavy assets.
"""

from backtest_lib.examples.hold import strategy as hold_strategy

__all__ = ["hold_strategy"]
