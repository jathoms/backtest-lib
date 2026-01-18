from backtest_lib import combine, hold, reallocate, target_weights, trade
from backtest_lib.engine.decision import CompositeDecision


def test_decision_combination():
    decision = trade("buy", 10, "security1") + target_weights({"security2": 0.5})
    assert isinstance(decision, CompositeDecision)
    assert len(decision.decisions) == 2


def test_composite_decision_combination():
    decision = trade("buy", 10, "security1") + target_weights({"security2": 0.5})
    comp_decision = decision + (
        trade("buy", 4, "security3")
        + reallocate(
            0.4, out_of=["security1", "security2"], into=["security3", "security4"]
        )
    )
    assert isinstance(comp_decision, CompositeDecision)
    assert len(comp_decision.decisions) == 4


def test_combine_decision():
    decision = combine(trade("buy", 2, "security1"), trade("buy", 2, "security2"))
    assert isinstance(decision, CompositeDecision)
    assert len(decision.decisions) == 2


def test_empty_combine_decision():
    decision = combine()
    assert decision == hold()
