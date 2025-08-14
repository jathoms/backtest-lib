# OBJECTIVE

## Overall

Maximise objective function (portfolio value, sharpe, etc) given a security universe.

### How?

## Trades

By executing the **trades** that will result in the highest objective value.

### How?

## Desired/Optimal Holdings

By finding the **optimal holdings** at any given time to trade towards according to our **strategy**.

### How?

## Data

By processing our data in such a way that produces **signals** that each contribute to our desired holdings.

maybe: _The contribution of each signal is what is optimized by the library in this problem_

# USER EXPERIENCE

If I want to use this library to evaluate my sick new strategy that came to me in a dream, what would be the perfect tool to easily check?

## 1. Buy and hold

The simplest strategy. Exactly one decision is made for holdings at the very start, and the performance of the portfolio rests solely on the performance of the initial holdings.

How might I want to create a backtest for it?

Here's a VERY ROUGH overview of what that could look like.

```python
universe = Universe(["stonk1", "stonk2"]) # Simple constructor to begin.
initial_holdings = Holdings(
    whatever,
    constructor,
    args,
)

@Strategy
def buy_and_hold_strategy(univ: Universe,
                          current_holdings: Holdings,
                          market_conditions: MarketConditions
                          current_period: CurrentPeriod) -> Holdings:
    return current_holdings

backtest = Backtest(buy_and_hold_strategy, initial_holdings)

stats: StrategyEval = backtest.run()
```
```python
from typing import Protocol, Mapping, Optional, Sequence, NamedTuple
import numpy as np
from dataclasses import dataclass

SecurityId = str  # or int

@dataclass(frozen=True)
class Universe:
    ids: Sequence[SecurityId]               # length N, stable order
    tradable: np.ndarray                    # shape (N,), bool
    meta: Mapping[str, np.ndarray] = None   # optional aligned fields (e.g., sector codes)

@dataclass(frozen=True)
class Holdings:
    weights: np.ndarray                     # shape (N,), sums to 1 incl. cash if present
    cash_weight: float = 0.0                # optional explicit cash

@dataclass(frozen=True)
class Period:
    index: int                              # e.g., bar number
    start_ns: int                           # epoch nanos
    end_ns: int
    freq: Frequency

@dataclass(frozen=True)
class MarketConditions:
    # All arrays are immutable, aligned to Universe.ids, and time-major where needed
    # Use views/slices to present only data up to `Period` (no look-ahead).
    close: np.ndarray                       # shape (T, N)
    volume: Optional[np.ndarray] = None     # shape (T, N)
    features: Optional[Mapping[str, np.ndarray]] = None  # e.g., "beta": (T,N), "alpha": (T,N)
    corporate_actions: Optional[Mapping[str, np.ndarray]] = None  # splits/divs masks
    risk_model: Optional[Mapping[str, np.ndarray]] = None         # factor loadings, cov

class Decision(NamedTuple):
    target: Holdings                         # desired end-of-period weights
    notes: Mapping[str, float] = {}          # diagnostics, scores, etc.

class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current: Holdings,
        market: MarketConditions,
        period: Period,
        *,
        seed: Optional[int] = None
    ) -> Decision: ...
```

```python
from typing import Callable

@dataclass(frozen=True)
class Constraints:
    max_turnover: float = 1.0               # fraction per rebalance
    max_weight: float = 0.1
    min_weight: float = 0.0
    long_only: bool = True
    blacklist: Optional[np.ndarray] = None  # shape (N,), bool

# Pure cost model: weights_now, weights_next -> estimated cost
CostFn = Callable[[np.ndarray, np.ndarray], float]

@dataclass(frozen=True)
class StrategyParams:
    k: Mapping[str, float]                  # free params, e.g., regularizers

class Decision(NamedTuple):
    target: Holdings
    notes: Mapping[str, float] = {}
    # optionally expose the *suggested* trades as a pure derivation:
    # trades = target.weights - current.weights (downstream can compute)

def strategy(
    universe: Universe,
    current: Holdings,
    market: MarketConditions,
    period: Period,
    *,
    constraints: Constraints = Constraints(),
    cost_fn: Optional[CostFn] = None,
    params: Optional[StrategyParams] = None,
    seed: Optional[int] = None,
) -> Decision:
    ...
```