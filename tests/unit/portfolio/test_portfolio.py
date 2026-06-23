import pytest

from backtest_lib.portfolio import (
    Cash,
    FractionalQuantityPortfolio,
    QuantityPortfolio,
    WeightedPortfolio,
    cash,
    uniform_portfolio,
)


@pytest.fixture()
def universe() -> tuple[str, ...]:
    return ("AAA", "BBB", "CCC")


@pytest.fixture()
def prices(universe: tuple[str, ...], mapping_type):
    return mapping_type.from_vectors(universe, [10.0, 20.0, 40.0])


def test_quantity_into_weighted_requires_prices(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    portfolio = QuantityPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [1, 2, 0]),
        cash=100.0,
        total_value=200.0,
        constructor_backend=market_backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_weighted()


def test_quantity_into_weighted(
    universe: tuple[str, ...], prices, market_backend: str, mapping_type
) -> None:
    portfolio = QuantityPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [1, 2, 0]),
        cash=100.0,
        total_value=200.0,
        constructor_backend=market_backend,
    )
    weighted = portfolio.into_weighted(prices)

    assert list(weighted.holdings.values()) == pytest.approx(
        [1.0 / 15.0, 4.0 / 15.0, 0.0]
    )
    assert weighted.cash == pytest.approx(2.0 / 3.0)
    assert weighted.total_value == pytest.approx(150.0)


def test_fractional_into_weighted_requires_prices(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 1.5, 0.0]),
        cash=10.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_weighted()


def test_fractional_into_weighted(
    universe: tuple[str, ...], prices, market_backend: str, mapping_type
) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 1.5, 0.0]),
        cash=10.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    weighted = portfolio.into_weighted(prices)

    assert list(weighted.holdings.values()) == pytest.approx(
        [1.0 / 9.0, 2.0 / 3.0, 0.0]
    )
    assert weighted.cash == pytest.approx(2.0 / 9.0)
    assert weighted.total_value == pytest.approx(45.0)


def test_fractional_into_quantities(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [1.9, 2.1, 0.5]),
        cash=5.0,
        total_value=50.0,
        constructor_backend=market_backend,
    )
    qty_portfolio = portfolio.into_quantities()
    assert list(qty_portfolio.holdings.values()) == [1, 2, 0]
    assert qty_portfolio.cash == pytest.approx(5.0)
    assert qty_portfolio.total_value == pytest.approx(50.0)


def test_weighted_into_quantities_requires_prices(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_quantities()


def test_weighted_into_quantities(
    universe: tuple[str, ...], prices, market_backend: str, mapping_type
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    qty_portfolio = portfolio.into_quantities(prices)
    assert list(qty_portfolio.holdings.values()) == [5, 1, 0]
    assert qty_portfolio.cash == pytest.approx(30.0)
    assert qty_portfolio.total_value == pytest.approx(100.0)


def test_weighted_into_quantities_fractional_requires_prices(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.25,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_quantities_fractional()


def test_weighted_into_quantities_fractional(
    universe: tuple[str, ...], prices, market_backend: str, mapping_type
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.25,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    fractional = portfolio.into_quantities_fractional(prices)
    assert list(fractional.holdings.values()) == pytest.approx([5.0, 1.25, 0.625])
    assert fractional.cash == pytest.approx(25.0)
    assert fractional.total_value == pytest.approx(100.0)


def test_weighted_into_long_only_invariants(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    if market_backend == "native":
        pytest.skip("native backend does not yet implement into_long_only")
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=mapping_type.from_vectors(universe, [0.6, -0.2, 0.6]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    long_only = portfolio.into_long_only()
    weights = list(long_only.holdings.values())
    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)
    assert long_only.cash == pytest.approx(0.0)


def test_weighted_indexed_over(
    universe: tuple[str, ...], market_backend: str, mapping_type
) -> None:
    base = WeightedPortfolio(
        universe=("AAA", "BBB"),
        holdings=mapping_type.from_vectors(("AAA", "BBB"), [0.6, 0.4]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=market_backend,
    )
    indexed = base.indexed_over(universe)
    assert list(indexed.holdings.values()) == pytest.approx([0.5, 0.5, 0.0])


def test_uniform_portfolio_full_universe(
    universe: tuple[str, ...], market_backend: str
) -> None:
    portfolio = uniform_portfolio(universe, value=10.0, backend=market_backend)
    assert list(portfolio.holdings.values()) == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    )
    assert portfolio.total_value == pytest.approx(10.0)


def test_uniform_portfolio_tradable_subset(
    universe: tuple[str, ...], market_backend: str
) -> None:
    portfolio = uniform_portfolio(
        universe, tradable_universe={"AAA", "CCC"}, backend=market_backend
    )
    assert list(portfolio.holdings.values()) == pytest.approx([0.5, 0.0, 0.5])


def test_uniform_portfolio_tradable_list(
    universe: tuple[str, ...], market_backend: str
) -> None:
    portfolio = uniform_portfolio(
        universe, tradable_universe=["AAA"], backend=market_backend
    )
    assert list(portfolio.holdings.values()) == pytest.approx([1.0, 0.0, 0.0])


def test_cash_materialize(universe: tuple[str, ...], market_backend: str) -> None:
    portfolio = cash(1000.0).materialize(universe, market_backend)
    assert list(portfolio.holdings.values()) == pytest.approx([0.0, 0.0, 0.0])
    assert portfolio.cash == pytest.approx(1.0)
    assert portfolio.total_value == pytest.approx(1000.0)


def test_cash_object_materialize(
    universe: tuple[str, ...], market_backend: str
) -> None:
    portfolio = Cash(250.0).materialize(universe, backend=market_backend)
    assert list(portfolio.holdings.values()) == pytest.approx([0.0, 0.0, 0.0])
    assert portfolio.cash == pytest.approx(1.0)
    assert portfolio.total_value == pytest.approx(250.0)
