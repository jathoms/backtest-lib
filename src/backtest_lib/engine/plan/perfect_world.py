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
    MakeTradesOp,
    Plan,
    TargetHoldingsOp,
    TargettingOp,
    TargetWeightsOp,
    TradeOrder,
    Trades,
)
from backtest_lib.market import get_mapping_type_from_mapping
from backtest_lib.universe.universe_mapping import UniverseMapping


class DuplicateTargetException(Exception): ...


PerfectWorldOps = TargetWeightsOp | TargetHoldingsOp | MakeTradesOp
_PerfectWorldAtomicOps = TargetWeightsOp | TargetHoldingsOp | MakeTradeOp


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
    ) -> Iterator[_PerfectWorldAtomicOps]:
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

    def _normalize_ops(
        self, atomic_ops: Iterable[_PerfectWorldAtomicOps]
    ) -> Iterator[PerfectWorldOps]:
        trades = []
        targetting_op: TargettingOp | None = None

        for op in atomic_ops:
            if isinstance(op, MakeTradeOp):
                trades.append(op.trade)
            elif isinstance(op, TargettingOp):
                if targetting_op is not None:
                    raise DuplicateTargetException()
                targetting_op = op
                yield op
            else:
                assert_never(op)
        batched_trades = Trades(
            trades=tuple(trades),
            security_alignment=self._security_alignment,
            backend_mapping_type=self._backend_mapping_type,
        )
        yield MakeTradesOp(batched_trades)

    def generate_plan(
        self, decision: Decision, prices: UniverseMapping
    ) -> Plan[PerfectWorldOps]:
        ops = self._normalize_ops(self._parse_decision(decision, prices))
        return Plan(steps=tuple(ops))
