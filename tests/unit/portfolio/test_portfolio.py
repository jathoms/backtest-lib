import pytest

from backtest_lib.market.polars_impl import PolarsUniverseMapping
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
def prices(universe: tuple[str, ...]) -> PolarsUniverseMapping[float]:
    return PolarsUniverseMapping.from_vectors(universe, [10.0, 20.0, 40.0])


@pytest.mark.parametrize("backend", ["polars"])
def test_quantity_into_weighted_requires_prices(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = QuantityPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [1, 2, 0]),
        cash=100.0,
        total_value=200.0,
        constructor_backend=backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_weighted()


@pytest.mark.parametrize("backend", ["polars"])
def test_quantity_into_weighted(
    universe: tuple[str, ...], prices: PolarsUniverseMapping[float], backend: str
) -> None:
    portfolio = QuantityPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [1, 2, 0]),
        cash=100.0,
        total_value=200.0,
        constructor_backend=backend,
    )
    weighted = portfolio.into_weighted(prices)

    assert list(weighted.holdings.values()) == pytest.approx(
        [1.0 / 15.0, 4.0 / 15.0, 0.0]
    )
    assert weighted.cash == pytest.approx(2.0 / 3.0)
    assert weighted.total_value == pytest.approx(150.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_fractional_into_weighted_requires_prices(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 1.5, 0.0]),
        cash=10.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_weighted()


@pytest.mark.parametrize("backend", ["polars"])
def test_fractional_into_weighted(
    universe: tuple[str, ...], prices: PolarsUniverseMapping[float], backend: str
) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 1.5, 0.0]),
        cash=10.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    weighted = portfolio.into_weighted(prices)

    assert list(weighted.holdings.values()) == pytest.approx(
        [1.0 / 9.0, 2.0 / 3.0, 0.0]
    )
    assert weighted.cash == pytest.approx(2.0 / 9.0)
    assert weighted.total_value == pytest.approx(45.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_fractional_into_quantities(universe: tuple[str, ...], backend: str) -> None:
    portfolio = FractionalQuantityPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [1.9, 2.1, 0.5]),
        cash=5.0,
        total_value=50.0,
        constructor_backend=backend,
    )
    qty_portfolio = portfolio.into_quantities()
    assert list(qty_portfolio.holdings.values()) == [1, 2, 0]
    assert qty_portfolio.cash == pytest.approx(5.0)
    assert qty_portfolio.total_value == pytest.approx(50.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_into_quantities_requires_prices(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_quantities()


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_into_quantities(
    universe: tuple[str, ...], prices: PolarsUniverseMapping[float], backend: str
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    qty_portfolio = portfolio.into_quantities(prices)
    assert list(qty_portfolio.holdings.values()) == [5, 1, 0]
    assert qty_portfolio.cash == pytest.approx(30.0)
    assert qty_portfolio.total_value == pytest.approx(100.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_into_quantities_fractional_requires_prices(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.25,
        total_value=100.0,
        constructor_backend=backend,
    )
    with pytest.raises(ValueError):
        portfolio.into_quantities_fractional()


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_into_quantities_fractional(
    universe: tuple[str, ...], prices: PolarsUniverseMapping[float], backend: str
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.5, 0.25, 0.25]),
        cash=0.25,
        total_value=100.0,
        constructor_backend=backend,
    )
    fractional = portfolio.into_quantities_fractional(prices)
    assert list(fractional.holdings.values()) == pytest.approx([5.0, 1.25, 0.625])
    assert fractional.cash == pytest.approx(25.0)
    assert fractional.total_value == pytest.approx(100.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_into_long_only_invariants(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = WeightedPortfolio(
        universe=universe,
        holdings=PolarsUniverseMapping.from_vectors(universe, [0.6, -0.2, 0.6]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    long_only = portfolio.into_long_only()
    weights = list(long_only.holdings.values())
    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)
    assert long_only.cash == pytest.approx(0.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_weighted_indexed_over(universe: tuple[str, ...], backend: str) -> None:
    base = WeightedPortfolio(
        universe=("AAA", "BBB"),
        holdings=PolarsUniverseMapping.from_vectors(("AAA", "BBB"), [0.6, 0.4]),
        cash=0.0,
        total_value=100.0,
        constructor_backend=backend,
    )
    indexed = base.indexed_over(universe)
    assert list(indexed.holdings.values()) == pytest.approx([0.5, 0.5, 0.0])


@pytest.mark.parametrize("backend", ["polars"])
def test_uniform_portfolio_full_universe(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = uniform_portfolio(universe, value=10.0, backend=backend)
    assert list(portfolio.holdings.values()) == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    )
    assert portfolio.total_value == pytest.approx(10.0)


@pytest.mark.parametrize("backend", ["polars"])
def test_uniform_portfolio_tradable_subset(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = uniform_portfolio(
        universe, tradable_universe={"AAA", "CCC"}, backend=backend
    )
    assert list(portfolio.holdings.values()) == pytest.approx([0.5, 0.0, 0.5])


@pytest.mark.parametrize("backend", ["polars"])
def test_uniform_portfolio_tradable_list(
    universe: tuple[str, ...], backend: str
) -> None:
    portfolio = uniform_portfolio(universe, tradable_universe=["AAA"], backend=backend)
    assert list(portfolio.holdings.values()) == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize("backend", ["polars"])
def test_cash_materialize(universe: tuple[str, ...], backend: str) -> None:
    portfolio = cash(1000.0).materialize(universe, backend)
    assert list(portfolio.holdings.values()) == pytest.approx([0.0, 0.0, 0.0])
    assert portfolio.cash == pytest.approx(1.0)
    assert portfolio.total_value == pytest.approx(1000.0)


def test_cash_object_materialize(universe: tuple[str, ...]) -> None:
    portfolio = Cash(250.0).materialize(universe, backend="polars")
    assert list(portfolio.holdings.values()) == pytest.approx([0.0, 0.0, 0.0])
    assert portfolio.cash == pytest.approx(1.0)
    assert portfolio.total_value == pytest.approx(250.0)
