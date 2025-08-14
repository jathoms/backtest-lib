import numpy as np


def assert_no_lookahead(strategy, builder, universe, current, periods):
    # audit: run twice, once normally and once with future rows scrambled
    decisions = []
    for p in periods:
        m1 = builder.make(universe, p, scramble_future=False)
        m2 = builder.make(universe, p, scramble_future=True)
        d1 = strategy(universe, current, m1, p, seed=123)
        d2 = strategy(universe, current, m2, p, seed=123)
        # Should match bit-for-bit (or within tolerance if FP noise)
        np.testing.assert_allclose(d1.target.weights, d2.target.weights, atol=0, rtol=0)
        decisions.append(d1)
        current = d1.target
    return decisions
