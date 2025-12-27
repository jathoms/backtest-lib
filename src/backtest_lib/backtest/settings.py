from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    """PLACEHOLDER"""

    allow_short: bool

    @staticmethod
    def default() -> BacktestSettings:
        """PLACEHOLDER"""
        return BacktestSettings(allow_short=False)
