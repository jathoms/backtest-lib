from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import assert_never

from backtest_lib.engine.decision import (
    CompositeDecision,
    Decision,
    HoldDecision,
    MakeTradeDecision,
    ReallocateDecision,
    TargetHoldingsDecision,
    TargetWeightsDecision,
)
from backtest_lib.engine.plan import (
    MakeTradeOp,
    Plan,
    ReallocateOp,
    TargetHoldingsOp,
    TargetWeightsOp,
    TradeOrder,
)
from backtest_lib.universe.universe_mapping import UniverseMapping

logger = logging.getLogger(__name__)

PerfectWorldOps = TargetWeightsOp | TargetHoldingsOp | MakeTradeOp | ReallocateOp


class PerfectWorldPlanGenerator:
    def _parse_decision(
        self, decision: Decision, prices: UniverseMapping
    ) -> Iterator[PerfectWorldOps]:
        if isinstance(decision, TargetWeightsDecision):
            yield TargetWeightsOp(
                weights=decision.target_weights,
                cash=decision.cash,
                fill_cash=decision.fill_cash,
            )
        elif isinstance(decision, TargetHoldingsDecision):
            yield TargetHoldingsOp(
                holdings=decision.target_holdings,
                cash=decision.cash,
                fill_cash=decision.fill_cash,
            )
        elif isinstance(decision, ReallocateDecision):
            yield ReallocateOp(decision)
        elif isinstance(decision, MakeTradeDecision):
            trade_order = TradeOrder(
                direction=decision.direction,
                qty=decision.qty,
                security=decision.security,
                price=prices[decision.security],
            )
            yield MakeTradeOp(trade_order)
        elif isinstance(decision, CompositeDecision):
            for step in decision.decisions:
                yield from self._parse_decision(step, prices)
        elif isinstance(decision, HoldDecision):
            return
        else:
            assert_never(decision)

    def generate_plan(
        self, decision: Decision, prices: UniverseMapping
    ) -> Plan[PerfectWorldOps]:
        ops = self._parse_decision(decision, prices)
        return Plan(steps=ops)
