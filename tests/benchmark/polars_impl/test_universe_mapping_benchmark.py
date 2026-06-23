"""Benchmarks for UniverseMapping ordering behavior.

These timings highlight the performance impact when key order differs between mappings.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from backtest_lib.market._backends import _get_mapping_type_from_backend


def _make_mappings(
    backend: str,
    keys: list[str],
    other_keys: list[str],
) -> tuple[object, object]:
    values = [1] * len(keys)
    mapping_type = _get_mapping_type_from_backend(backend)
    return (
        mapping_type.from_vectors(keys, values),
        mapping_type.from_vectors(other_keys, values),
    )


def _benchmark_construction(
    backend: str,
    keys: list[str],
    other_keys: list[str],
) -> object:
    acc, other = _make_mappings(backend, keys, other_keys)
    return cast(Any, acc) + cast(Any, other)


def _benchmark_addition(acc: object, other: object) -> object:
    return cast(Any, acc) + cast(Any, other)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_small_ordering_same(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = ["a", "b", "c"]
    acc, other = _make_mappings(backend, keys, keys)
    benchmark(_benchmark_addition, acc, other)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_small_ordering_diff(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = ["a", "b", "c"]
    diff_keys = ["c", "a", "b"]
    acc, other = _make_mappings(backend, keys, diff_keys)
    benchmark(_benchmark_addition, acc, other)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_large_ordering_same(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = [str(i) for i in range(1000)]
    acc, other = _make_mappings(backend, keys, keys)
    benchmark(_benchmark_addition, acc, other)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_large_ordering_diff(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = [str(i) for i in range(1000)]
    diff_keys = list(reversed(keys))
    acc, other = _make_mappings(backend, keys, diff_keys)
    benchmark(_benchmark_addition, acc, other)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_large_construction_same(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = [str(i) for i in range(1000)]
    benchmark(_benchmark_construction, backend, keys, keys)


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", ["polars", "native"])
def test_large_construction_diff(benchmark: BenchmarkFixture, backend: str) -> None:
    keys = [str(i) for i in range(1000)]
    diff_keys = list(reversed(keys))
    benchmark(_benchmark_construction, backend, keys, diff_keys)
