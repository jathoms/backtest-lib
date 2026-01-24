"""Benchmarks for UniverseMapping ordering behavior.

These timings highlight the performance impact when key order differs between mappings.
"""

from __future__ import annotations

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from backtest_lib.market.polars_impl import PolarsUniverseMapping


def _benchmark_addition(
    keys: list[str],
    other_keys: list[str],
) -> PolarsUniverseMapping:
    values = [1] * len(keys)
    acc = PolarsUniverseMapping.from_vectors(keys, values)
    other = PolarsUniverseMapping.from_vectors(other_keys, values)
    return acc + other


@pytest.mark.benchmark
def test_small_ordering_same(benchmark: BenchmarkFixture) -> None:
    keys = ["a", "b", "c"]
    benchmark(_benchmark_addition, keys, keys)


@pytest.mark.benchmark
def test_small_ordering_diff(benchmark: BenchmarkFixture) -> None:
    keys = ["a", "b", "c"]
    diff_keys = ["c", "a", "b"]
    benchmark(_benchmark_addition, keys, diff_keys)


@pytest.mark.benchmark
def test_large_ordering_same(benchmark: BenchmarkFixture) -> None:
    keys = [str(i) for i in range(1000)]
    benchmark(_benchmark_addition, keys, keys)


@pytest.mark.benchmark
def test_large_ordering_diff(benchmark: BenchmarkFixture) -> None:
    keys = [str(i) for i in range(1000)]
    diff_keys = list(reversed(keys))
    benchmark(_benchmark_addition, keys, diff_keys)
