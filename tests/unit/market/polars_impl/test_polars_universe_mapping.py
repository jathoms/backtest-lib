import operator

import polars as pl
import pytest

from backtest_lib.market.polars_impl._universe_mapping import PolarsUniverseMapping


@pytest.fixture()
def names() -> tuple[str, ...]:
    return ("AAA", "BBB", "CCC")


@pytest.mark.parametrize(
    ("values", "expected", "expected_type"),
    [
        ([1, 2, 3], [1, 2, 3], int),
        ([1.5, 2.0, 3.25], [1.5, 2.0, 3.25], float),
        (lambda: (x for x in [1, 2, 3]), [1, 2, 3], int),
    ],
)
def test_from_vectors_values(
    names: tuple[str, ...],
    values,
    expected: list[float | int],
    expected_type: type[int] | type[float],
) -> None:
    data = values() if callable(values) else values
    mapping = PolarsUniverseMapping.from_vectors(names, data)  # type: ignore[arg-type]
    assert mapping.to_series().to_list() == expected
    assert isinstance(mapping["AAA"], expected_type)


def test_from_names_and_data_length_mismatch(names: tuple[str, ...]) -> None:
    data = pl.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        PolarsUniverseMapping.from_names_and_data(names, data)


def test_getitem_list_order(names: tuple[str, ...]) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [1.0, 2.0, 3.0])
    result = mapping[["CCC", "AAA"]]
    assert result.to_list() == [3.0, 1.0]


@pytest.mark.parametrize("key", [1, ("AAA", "BBB")])
def test_getitem_invalid_type(names: tuple[str, ...], key) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    with pytest.raises(ValueError):
        mapping[key]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("op", "scalar", "expected", "expected_type"),
    [
        (operator.add, 1, [2, 3, 4], int),
        (operator.sub, 1, [0, 1, 2], int),
        (operator.mul, 2, [2, 4, 6], int),
        (operator.truediv, 2, [0.5, 1.0, 1.5], float),
    ],
)
def test_scalar_arithmetic_ops(
    names: tuple[str, ...],
    op,
    scalar: int,
    expected: list[float | int],
    expected_type: type[int] | type[float],
) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    result = op(mapping, scalar)
    assert result.to_series().to_list() == expected
    assert result._scalar_type is expected_type


def test_scalar_arithmetic_promotion(names: tuple[str, ...]) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    mixed = mapping + 1.5
    assert mixed.to_series().to_list() == [2.5, 3.5, 4.5]
    assert mixed._scalar_type is float


def test_mapping_arithmetic_and_narrowing(names: tuple[str, ...]) -> None:
    base = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    other = PolarsUniverseMapping.from_vectors(names, [0.5, 1.0, 1.5])
    combined = base + other
    assert combined.to_series().to_list() == [1.5, 3.0, 4.5]
    assert combined._scalar_type is float


def test_mapping_subset_alignment(names: tuple[str, ...]) -> None:
    base = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    other = PolarsUniverseMapping.from_vectors(("AAA", "CCC"), [10, 30])
    combined = base + other
    assert combined.to_series().to_list() == [11, 2, 33]


def test_mapping_mismatched_universe_raises(names: tuple[str, ...]) -> None:
    base = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    other = PolarsUniverseMapping.from_vectors(("AAA", "DDD"), [1, 2])
    with pytest.raises(TypeError):
        _ = base + other


def test_dict_alignment(names: tuple[str, ...]) -> None:
    base = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    combined = base + {"AAA": 10}
    assert combined.to_series().to_list() == [11, 2, 3]


def test_dict_alignment_promotes_float(names: tuple[str, ...]) -> None:
    base = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    combined = base + {"AAA": 1.5}
    assert combined.to_series().to_list() == [2.5, 2, 3]
    assert combined._scalar_type is float


@pytest.mark.parametrize(
    ("op", "scalar", "expected", "expected_type"),
    [
        (operator.add, 2, [3, 4, 5], int),
        (operator.sub, 2, [1, 0, -1], int),
        (operator.mul, 2, [2, 4, 6], int),
        (operator.truediv, 2, [2.0, 1.0, 2 / 3], float),
    ],
)
def test_reflected_scalar_ops(
    names: tuple[str, ...],
    op,
    scalar: int,
    expected: list[float | int],
    expected_type: type[int] | type[float],
) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [1, 2, 3])
    result = op(scalar, mapping)
    if expected_type is float:
        assert result.to_series().to_list() == pytest.approx(expected)
    else:
        assert result.to_series().to_list() == expected
    assert result._scalar_type is expected_type


def test_sum_mean_abs_floor_truncate(names: tuple[str, ...]) -> None:
    mapping = PolarsUniverseMapping.from_vectors(names, [-1.2, 2.7, -3.1])
    assert mapping.sum() == pytest.approx(-1.6)
    assert mapping.mean() == pytest.approx(-0.5333333333333333)
    assert mapping.abs().to_series().to_list() == [1.2, 2.7, 3.1]
    assert mapping.floor().to_series().to_list() == [-2, 2, -4]
    assert mapping.floor()._scalar_type is int
    assert mapping.truncate().to_series().to_list() == [-1, 2, -3]
    assert mapping.truncate()._scalar_type is int


def test_mean_on_non_numeric_raises(names: tuple[str, ...]) -> None:
    data = pl.Series(["a", "b", "c"])
    mapping = PolarsUniverseMapping.from_names_and_data(names, data)
    with pytest.raises(TypeError):
        mapping.mean()


def test_mean_empty_series_raises() -> None:
    mapping = PolarsUniverseMapping.from_names_and_data(
        (), pl.Series([], dtype=pl.Float64)
    )
    with pytest.raises(ValueError):
        mapping.mean()
