import pytest

from backtest_lib.engine.decision import (
    CompositeDecision,
    TradeDirection,
    combine,
    hold,
    reallocate,
    target_holdings,
    target_weights,
    trade,
)
from backtest_lib.engine.plan import (
    MakeTradeOp,
    Plan,
    ReallocateOp,
    TargetHoldingsOp,
    TargetWeightsOp,
    TradeOrder,
)
from backtest_lib.engine.plan.perfect_world import PerfectWorldPlanGenerator
from backtest_lib.universe.universe_mapping import make_universe_mapping


def _prices(mapping: dict[str, float], *, universe: list[str] | None = None):
    """
    Helper to build a UniverseMapping[float] for prices.
    """
    if universe is None:
        universe = list(mapping.keys())
    return make_universe_mapping(mapping, universe)


def test_target_weights_decision_emits_target_weights_op():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 100.0})

    decision = target_weights({"A": 1.0}, fill_cash=True)

    ops = list(gen._parse_decision(decision, prices))
    assert ops == [
        TargetWeightsOp(weights={"A": 1.0}, cash=None, fill_cash=True),
    ]


def test_target_holdings_decision_emits_target_holdings_op():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 100.0})

    decision = target_holdings({"A": 3}, fill_cash=False)

    ops = list(gen._parse_decision(decision, prices))
    assert ops == [
        TargetHoldingsOp(holdings={"A": 3}, cash=None, fill_cash=False),
    ]


def test_price_is_injected_correctly_to_make_trade():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 101.25})

    decision = trade("buy", 3, "A")

    ops = list(gen._parse_decision(decision, prices))
    assert len(ops) == 1
    assert isinstance(ops[0], MakeTradeOp)
    assert ops[0].trade == TradeOrder(
        direction=TradeDirection.BUY,
        qty=3,
        security="A",
        price=101.25,
    )


def test_make_trade_decision_missing_price_raises_keyerror():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0}, universe=["A"])

    decision = trade("sell", 1, "B")

    with pytest.raises(KeyError):
        list(gen._parse_decision(decision, prices))


def test_reallocate_decision_emits_reallocate_op_wrapping_same_decision():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0, "B": 20.0})

    decision = reallocate(0.1, out_of=["A"], into=["B"], mode="equal_out_equal_in")

    ops = list(gen._parse_decision(decision, prices))
    assert len(ops) == 1
    assert isinstance(ops[0], ReallocateOp)
    assert ops[0].inner is decision


def test_hold_decision_emits_no_ops():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0})

    decision = hold()

    ops = list(gen._parse_decision(decision, prices))
    assert ops == []


def test_composite_decision_flattens_in_order_and_skips_hold():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0, "B": 20.0})

    d1 = target_weights({"A": 1.0}, fill_cash=True)
    d2 = hold()
    d3 = trade("sell", 2, "B")
    d4 = target_holdings({"A": 5}, fill_cash=False)

    decision = combine(d1, d2, d3, d4)

    ops = list(gen._parse_decision(decision, prices))
    assert [type(op) for op in ops] == [TargetWeightsOp, MakeTradeOp, TargetHoldingsOp]
    assert ops[0] == TargetWeightsOp(weights={"A": 1.0}, cash=None, fill_cash=True)
    assert ops[1] == MakeTradeOp(
        trade=TradeOrder(direction=TradeDirection.SELL, qty=2, security="B", price=20.0)
    )
    assert ops[2] == TargetHoldingsOp(holdings={"A": 5}, cash=None, fill_cash=False)


def test_nested_composites_are_flattened_depth_first():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0, "B": 20.0})

    inner = CompositeDecision(
        decisions=(
            target_holdings({"A": 1}),
            trade("buy", 1, "B"),
        )
    )
    outer = CompositeDecision(
        decisions=(
            target_weights({"A": 1.0}),
            inner,
            hold(),
            target_holdings({"B": 2}),
        )
    )

    ops = list(gen._parse_decision(outer, prices))
    assert [type(op) for op in ops] == [
        TargetWeightsOp,
        TargetHoldingsOp,
        MakeTradeOp,
        TargetHoldingsOp,
    ]


def test_generate_plan_materializes_steps_as_tuple():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0})

    decision = combine(
        target_weights({"A": 1.0}),
        trade("buy", 1, "A"),
    )

    plan = gen.generate_plan(decision, prices)
    assert isinstance(plan, Plan)
    assert isinstance(plan.steps, tuple)
    assert len(plan.steps) == 2
    assert isinstance(plan.steps[0], TargetWeightsOp)
    assert isinstance(plan.steps[1], MakeTradeOp)


def test_unknown_decision_triggers_assert_never():
    gen = PerfectWorldPlanGenerator()
    prices = _prices({"A": 10.0})

    class BogusDecision:
        pass

    with pytest.raises(AssertionError):
        list(gen._parse_decision(BogusDecision(), prices))  # type: ignore[arg-type]
