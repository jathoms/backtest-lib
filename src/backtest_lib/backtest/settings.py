from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    allow_short: bool

    @staticmethod
    def default() -> BacktestSettings:
        return BacktestSettings(allow_short=False)
