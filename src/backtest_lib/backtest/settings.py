from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    """Configuration for backtest execution constraints.

    Attributes:
        allow_short: Whether strategies may target negative weights.

    """

    allow_short: bool

    @staticmethod
    def default() -> BacktestSettings:
        """Return the default backtest settings.

        Returns:
            BacktestSettings with conservative defaults (shorting disabled).

        """
        return BacktestSettings(allow_short=False)
