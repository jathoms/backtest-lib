from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# CostFn = Callable[[np.ndarray, np.ndarray], float]


class HasParams(Protocol):
    @property
    def params(self) -> Mapping[str, float]: ...


class HasConstraints(Protocol):
    @property
    def max_weight(self) -> float: ...
    @property
    def long_only(self) -> bool: ...
    @property
    def max_turnover(self) -> float: ...


class HasRng(Protocol):
    def rng(self, seed: int | None) -> np.random.Generator: ...


@dataclass(frozen=True)
class StrategyContext(HasParams, HasConstraints, HasRng):
    params: Mapping[str, float]
    max_weight: float
    long_only: bool
    max_turnover: float

    # cost_fn: Optional[CostFn] = None
    rng_factory: Callable[[int | None], np.random.Generator] = (
        lambda seed: np.random.default_rng(seed)
    )

    def rng(self, seed: int | None) -> np.random.Generator:
        return self.rng_factory(seed)
