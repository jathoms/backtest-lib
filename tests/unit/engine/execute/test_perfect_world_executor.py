from typing import cast

import pytest

from backtest_lib.engine.decision import (
    ReallocationMode,
    TradeDirection,
    reallocate,
    target_weights,
    trade,
)
from backtest_lib.engine.execute.perfect_world import (
    DuplicateTargetException,
    NegativeCashException,
    PerfectWorldPlanExecutor,
    Trades,
    _TargetHoldingsCompiledOp,
    _TargetWeightsCompiledOp,
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
from backtest_lib.market import MarketView
from backtest_lib.portfolio import QuantityPortfolio, WeightedPortfolio
from backtest_lib.universe.universe_mapping import make_universe_mapping


def _prices(
    mapping: dict[str, float], *, universe: tuple[str, ...], backend: str = "polars"
):
    return make_universe_mapping(
        mapping, universe=universe, constructor_backend=backend
    )


def _weighted_pf(
    *,
    universe: tuple[str, ...],
    weights: dict[str, float],
    cash: float,
    total_value: float = 1000.0,
    backend: str = "polars",
):
    w = make_universe_mapping(weights, universe=universe, constructor_backend=backend)
    return WeightedPortfolio(
        universe=universe,
        holdings=w,
        cash=cash,
        total_value=total_value,
        constructor_backend=backend,
    )


def _quantity_pf(
    *,
    universe: tuple[str, ...],
    holdings: dict[str, int],
    cash: float,
    total_value: float = 1000.0,
    backend: str = "polars",
):
    h = make_universe_mapping(holdings, universe=universe, constructor_backend=backend)
    return QuantityPortfolio(
        universe=universe,
        holdings=h,
        cash=cash,
        total_value=total_value,
        constructor_backend=backend,
    )


def test_normalize_ops_yields_target_first_then_trades():
    universe = ("A", "B")
    backend = "polars"

    prices = _prices({"A": 10.0, "B": 20.0}, universe=universe, backend=backend)
    pf = _weighted_pf(
        universe=universe, weights={"A": 0.5, "B": 0.5}, cash=0.0, backend=backend
    )

    gen = PerfectWorldPlanGenerator()
    decision = target_weights({"A": 1.0}, fill_cash=True) + trade("buy", 1, "B")
    plan = gen.generate_plan(decision, prices)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    compiled = list(ex._normalize_ops(plan.steps, pf, prices))

    assert isinstance(compiled[0], _TargetWeightsCompiledOp)
    assert compiled[0].weights["A"] == 1.0
    assert compiled[0].cash == pytest.approx(0.0)

    # Trades (if present) must come after targetting op
    assert compiled[-1].__class__.__name__ == "Trades"
    assert len(compiled) == 2


def test_duplicate_target_raises():
    universe = ("A",)
    backend = "polars"
    prices = _prices({"A": 10.0}, universe=universe, backend=backend)
    pf = _weighted_pf(universe=universe, weights={"A": 1.0}, cash=0.0, backend=backend)

    ops = (
        TargetWeightsOp(weights={"A": 1.0}, cash=None, fill_cash=True),
        TargetHoldingsOp(holdings={"A": 1}, cash=None, fill_cash=False),
    )

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)

    with pytest.raises(DuplicateTargetException):
        list(ex._normalize_ops(ops, pf, prices))


def test_target_weights_fill_cash_true_with_manual_cash_warns_and_ignores_cash_value():
    universe = ("A", "B")
    backend = "polars"
    prices = _prices({"A": 10.0, "B": 10.0}, universe=universe, backend=backend)
    pf = _weighted_pf(
        universe=universe, weights={"A": 0.5, "B": 0.5}, cash=0.0, backend=backend
    )

    op = TargetWeightsOp(weights={"A": 0.7}, cash=999.0, fill_cash=True)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)

    with pytest.warns(UserWarning, match="fill_cash is set to True.*will be ignored"):
        compiled = list(ex._normalize_ops((op,), pf, prices))

    assert len(compiled) == 1
    assert isinstance(compiled[0], _TargetWeightsCompiledOp)
    assert compiled[0].weights["A"] == pytest.approx(0.7)
    assert compiled[0].cash == pytest.approx(0.3)


def test_target_holdings_fill_cash_true_sets_cash_to_preserve_total_value():
    universe = ("A", "B")
    backend = "polars"
    prices = _prices({"A": 10.0, "B": 5.0}, universe=universe, backend=backend)

    pf = _quantity_pf(
        universe=universe,
        holdings={"A": 0, "B": 0},
        cash=100.0,
        total_value=100.0,
        backend=backend,
    )

    # holdings value = 3*10 + 4*5 = 30 + 20 = 50 => cash should become 50
    op = TargetHoldingsOp(holdings={"A": 3, "B": 4}, cash=None, fill_cash=True)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    compiled = list(ex._normalize_ops((op,), pf, prices))

    assert len(compiled) == 1
    assert isinstance(compiled[0], _TargetHoldingsCompiledOp)
    assert compiled[0].holdings["A"] == 3
    assert compiled[0].holdings["B"] == 4
    assert compiled[0].cash == pytest.approx(50.0)


def test_reallocate_without_explicit_target_uses_portfolio_as_base_weighted():
    universe = ("A", "B")
    backend = "polars"
    prices = _prices({"A": 10.0, "B": 10.0}, universe=universe, backend=backend)

    pf = _weighted_pf(
        universe=universe, weights={"A": 1.0, "B": 0.0}, cash=0.0, backend=backend
    )

    decision = reallocate(
        0.2,
        out_of=["A"],
        into=["B"],
        mode=ReallocationMode.EQUAL_OUT_EQUAL_IN,
    )
    op = ReallocateOp(inner=decision)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    compiled = list(ex._normalize_ops((op,), pf, prices))

    assert len(compiled) == 1
    assert isinstance(compiled[0], _TargetWeightsCompiledOp)
    assert compiled[0].weights["A"] == pytest.approx(0.8)
    assert compiled[0].weights["B"] == pytest.approx(0.2)
    assert compiled[0].cash == pytest.approx(0.0)


def test_reallocate_pro_rata_out_equal_in_uses_position_values():
    universe = ("A", "B", "C", "D")
    backend = "polars"

    prices = make_universe_mapping(
        {"A": 20.0, "B": 10.0, "C": 5.0, "D": 5.0},
        universe=universe,
        constructor_backend=backend,
    )

    holdings = make_universe_mapping(
        {"A": 0.4, "B": 0.2, "C": 0.1, "D": 0.3},
        universe=universe,
        constructor_backend=backend,
    )
    pf = WeightedPortfolio(
        universe=universe,
        holdings=holdings,
        cash=0.0,
        total_value=1000.0,
        constructor_backend=backend,
    )

    decision = reallocate(
        0.3,
        out_of=["A", "B"],
        into=["C", "D"],
        mode=ReallocationMode.PRO_RATA_OUT_EQUAL_IN,
    )
    op = ReallocateOp(inner=decision)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    compiled = list(ex._normalize_ops((op,), pf, prices))

    assert len(compiled) == 1
    assert isinstance(compiled[0], _TargetWeightsCompiledOp)

    # expected pro-rata outflows:
    # A: 0.4*20 = 8
    # B: 0.2*10 = 2
    # total_from = 10
    # outflow weights: A = -(8/10)*0.3 = -0.24 ; B = -(2/10)*0.3 = -0.06
    # inflow to C and D = (1/2)*0.3 = 0.15 (equal-in)
    expected_A = pf.holdings["A"] - 0.24
    expected_B = pf.holdings["B"] - 0.06
    expected_C = pf.holdings["C"] + 0.15
    expected_D = pf.holdings["D"] + 0.15

    assert compiled[0].weights["A"] == pytest.approx(expected_A)
    assert compiled[0].weights["B"] == pytest.approx(expected_B)
    assert compiled[0].weights["C"] == pytest.approx(expected_C)
    assert compiled[0].weights["D"] == pytest.approx(expected_D)
    assert compiled[0].cash == pytest.approx(0.0)

    total = compiled[0].weights.sum() + compiled[0].cash
    assert total == pytest.approx(1.0)


def test_execute_plan_make_trade_updates_quantities_and_cash():
    universe = ("A",)
    backend = "polars"
    prices = _prices({"A": 10.0}, universe=universe, backend=backend)

    pf = _quantity_pf(
        universe=universe,
        holdings={"A": 2},
        cash=100.0,
        total_value=102.0,
        backend=backend,
    )

    op = MakeTradeOp(
        trade=TradeOrder(direction=TradeDirection.BUY, qty=3, security="A", price=10.0)
    )
    plan = Plan(steps=(op,))

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    result = ex.execute_plan(plan, pf, prices, market=cast(MarketView, None))

    after = result.after
    assert isinstance(after, QuantityPortfolio)
    assert after.holdings["A"] == 5
    assert after.cash == pytest.approx(70.0)


def test_execute_plan_raises_negative_cash_exception():
    universe = ("A",)
    backend = "polars"
    prices = _prices({"A": 10.0}, universe=universe, backend=backend)

    pf = _quantity_pf(
        universe=universe, holdings={"A": 0}, cash=5.0, total_value=5.0, backend=backend
    )

    plan = Plan(
        steps=(
            MakeTradeOp(
                trade=TradeOrder(
                    direction=TradeDirection.BUY, qty=1, security="A", price=5.1
                )
            ),
        )
    )

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)

    with pytest.raises(NegativeCashException):
        ex.execute_plan(plan, pf, prices, market=cast(MarketView, None))


def test_trades_position_delta_batches_by_security_and_respects_signed_qty():
    orders = (
        TradeOrder(direction=TradeDirection.BUY, qty=1, security="A", price=10.0),
        TradeOrder(direction=TradeDirection.BUY, qty=3, security="A", price=10.0),
        TradeOrder(direction=TradeDirection.SELL, qty=1, security="A", price=10.0),
        TradeOrder(direction=TradeDirection.SELL, qty=2, security="B", price=5.0),
    )
    universe = ("A", "B", "C")

    t = Trades.from_inputs(orders, security_alignment=universe, backend="polars")

    delta = t.position_delta

    expected = make_universe_mapping(
        {"A": 3, "B": -2, "C": 0},
        universe=universe,
        constructor_backend="polars",
    )

    assert all(
        k1 == k2 and v1 == v2
        for ((k1, v1), (k2, v2)) in zip(delta.items(), expected.items(), strict=True)
    )


def test_trades_position_delta_is_cached_property():
    orders = (
        TradeOrder(direction=TradeDirection.BUY, qty=1, security="A", price=10.0),
        TradeOrder(direction=TradeDirection.SELL, qty=1, security="B", price=5.0),
    )
    universe = ("A", "B")

    t = Trades.from_inputs(orders, security_alignment=universe, backend="polars")

    d1 = t.position_delta
    d2 = t.position_delta

    assert d1 is d2


def test_trades_total_cost_sums_signed_costs():
    orders = (
        TradeOrder(direction=TradeDirection.BUY, qty=2, security="A", price=10.0),
        TradeOrder(direction=TradeDirection.SELL, qty=3, security="B", price=5.0),
    )
    t = Trades.from_inputs(orders, security_alignment=("A", "B"), backend="polars")

    assert t.total_cost() == pytest.approx(20.0 - 15.0)


def test_trades_with_universe_returns_new_instance_and_recomputes_position_delta():
    orders = (
        TradeOrder(direction=TradeDirection.BUY, qty=1, security="A", price=10.0),
    )
    t = Trades.from_inputs(orders, security_alignment=("A",), backend="polars")

    orig_delta = t.position_delta

    t2 = t.with_universe(("A", "B"))

    assert t2 is not t
    assert t2.security_alignment == ("A", "B")
    assert t2.trades == t.trades
    assert t2.backend_mapping_type is t.backend_mapping_type

    delta2 = t2.position_delta
    assert delta2 is not orig_delta
    assert delta2["A"] == 1
    assert delta2["B"] == 0


def test_normalize_ops_target_holdings_weights_reallocation_converts_and_adjusts_cash():
    universe = ("A", "B")
    backend = "polars"

    # Pick prices that cause truncation loss:
    # total_value = 1000
    # reallocation fraction = 0.1 from A into B (equal out/in)
    # target_value_changes: A -100, B +100
    # qty_changes: A (-100/7) -> -14 (truncate), B (100/11) -> 9
    # actual_value_changes: A -14*7 = -98, B 9*11 = 99, sum = 1
    # cash_delta = -1, so cash decreases by 1.
    prices = make_universe_mapping(
        {"A": 7.0, "B": 11.0},
        universe=universe,
        constructor_backend=backend,
    )

    # Portfolio total_value is used by the conversion math.
    # Start with a quantity portfolio; holdings don't really matter for this branch,
    # but keep it simple.
    pf = QuantityPortfolio(
        universe=universe,
        holdings=make_universe_mapping(
            {"A": 0, "B": 0},
            universe=universe,
            constructor_backend=backend,
        ),
        cash=1000.0,
        total_value=1000.0,
        constructor_backend=backend,
    )

    # Target holdings op => compiled_targetting_op is _TargetHoldingsCompiledOp
    target_op = TargetHoldingsOp(
        holdings={"A": 0, "B": 0},
        cash=None,
        fill_cash=True,
        # makes initial cash exactly pf_total_value - holdings_value = 1000
    )

    # ReallocateOp in PRO-RATA vs EQUAL doesn't matter for this specific branch;
    # we just need a weights reallocation. EQUAL is easiest and deterministic.
    decision = reallocate(
        0.1,
        out_of=["A"],
        into=["B"],
        mode=ReallocationMode.EQUAL_OUT_EQUAL_IN,
    )
    realloc_op = ReallocateOp(inner=decision)

    ex = PerfectWorldPlanExecutor(backend=backend, security_alignment=universe)
    compiled = list(ex._normalize_ops((target_op, realloc_op), pf, prices))

    assert len(compiled) == 1
    assert isinstance(compiled[0], _TargetHoldingsCompiledOp)

    # Holdings changed by qty_changes computed via truncate():
    # A: -14, B: +9
    assert compiled[0].holdings["A"] == -14
    assert compiled[0].holdings["B"] == 9

    # Cash adjusted by cash_delta = -(actual_value_changes.sum()) = -1
    assert compiled[0].cash == pytest.approx(999.0)
