from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import assert_never

from backtest_lib.engine.decision import (
    CompositeDecision,
    Decision,
    HoldDecision,
    MakeTradeDecision,
    TargetHoldingsDecision,
    TargetWeightsDecision,
)
from backtest_lib.engine.plan import (
    MakeTradeOp,
    Plan,
    TargetHoldingsOp,
    TargetWeightsOp,
    TradeOrder,
)
from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping

PerfectWorldOps = TargetWeightsOp | TargetHoldingsOp | MakeTradeOp


class PerfectWorldPlanGenerator:
    _backend: str
    _security_alignment: tuple[str, ...]

    def __init__(self, backend: str, security_alignment: Iterable[str]):
        self._backend = backend
        self._security_alignment = tuple(security_alignment)

    @cached_property
    def _backend_mapping_type(self) -> type[UniverseMapping]:
        return get_mapping_type_from_mapping(self._backend)

    def _parse_decision(
        self, decision: Decision, prices: UniverseMapping
    ) -> Iterator[PerfectWorldOps]:
        if isinstance(decision, TargetWeightsDecision):
            yield TargetWeightsOp(decision.target_weights, decision.cash)
        elif isinstance(decision, TargetHoldingsDecision):
            yield TargetHoldingsOp(decision.target_holdings, decision.cash)
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
