from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Protocol, Self, assert_never

from backtest_lib.engine.decision import (
    CompositeDecision,
    Decision,
    DecisionT,
    HoldDecision,
    MakeTradeDecision,
    TargetHoldingsDecision,
    TargetWeightsDecision,
    TradeDirection,
)
from backtest_lib.engine.plan import (
    MakeTradeOp,
    MakeTradesOp,
    TargetHoldingsOp,
    TargetWeightsOp,
    TradeOrder,
    Trades,
)
from backtest_lib.market import MarketView, get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping

if TYPE_CHECKING:
    from backtest_lib.engine.plan import Plan, PlanOp


class DuplicateTargetException(Exception): ...


type PerfectWorldOps = TargetWeightsOp | TargetHoldingsOp | MakeTradesOp | MakeTradeOp


class PerfectWorldPlanGenerator:
    _backend_mapping_type: type[UniverseMapping]
    _security_alignment: tuple[str, ...]

    def _parse_decision(
        self, decision: DecisionT, prices: UniverseMapping
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
        self, decision: DecisionT, prices: UniverseMapping
    ) -> Plan[PerfectWorldOps]:
        trades = []
        ops: list[PerfectWorldOps] = []
        target_set = False
        for step in self._parse_decision(decision, prices):
            if isinstance(step, MakeTradeOp):
                trades.append(step.trade)
            elif isinstance(step, MakeTradesOp):
                trades.extend(step.trades.trades)
            elif isinstance(step, TargetHoldingsDecision | TargetWeightsDecision):
                if target_set:
                    raise DuplicateTargetException()
                ops.append(step)
                target_set = True
            else:
                ops.append(step)
        batched_trades = Trades(
            trades=tuple(trades),
            security_alignment=self._security_alignment,
            backend_mapping_type=self._backend_mapping_type,
        )
        ops.append(MakeTradesOp(trades=batched_trades))

        return Plan(steps=tuple(ops))
