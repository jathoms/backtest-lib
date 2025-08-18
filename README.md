# backtest-lib (name in progress)

## Usage (wip)

### Strategy

This library provides a lightweight framework for backtesting trading strategies. At its core, you define a strategy as a simple Python function that maps the current market state and portfolio into a decision about what to hold next. The library handles the rest: simulating trades over time, applying your decision rules at a fixed frequency, and generating performance statistics.

A strategy must conform to the following structure:

```python
class Decision:
    target: Holdings
    notes: UniverseMapping[Any] | None

class Strategy(Protocol):
    def __call__(
        self,
        universe: Universe,
        current_holdings: Holdings,
        market: MarketView,
        ctx: StrategyContext,
    ) -> Decision: ...
```

As inputs, the strategy receives the available universe, current holdings, a view of the market, and a context object for state.

For outputs, the strategy emits a Decision specifying the new target holdings for each decision point in the _decision schedule_ (TODO) (plus optional notes for inspection/memoized signals).

An toy example strategy can be written as follows, where we allocate our holdings uniformly across the all tradable assets:

```python
def equal_weight_strategy(
    universe: Universe,
    current_holdings: Holdings,
    market: MarketView,
    ctx: StrategyContext,
) -> Decision:
    weight = 1.0 / n
    target_holdings = Holdings({asset: weight for asset in universe})
    return Decision(target_holdings)
```

## Market

Inside the strategy function, the main way to interact with market data is through the MarketView object. This object provides a time-fenced view of historical prices, volumes, and tradability up to the current decision point. The data is time-fenced so that the strategy only sees information available at each step, as it marches forward through periods to reduce the risk of lookahead bias.

### Key properties:

- market.prices: access to OHLC price histories

- market.volume: access to per-security volume histories

- market.tradable: access to masks indicating which securities were tradable

Each of these is a PastView, which means we can:

Access the latest snapshot of close prices with `market.prices.close.by_period[-1]`,

access the data for only the last 5 periods with `market.prices.close.by_period[-5:]`,

access a single security’s full history with `market.prices.close.by_security["AAPL"]`,

or restrict the view to a time window with `market.volume.after(ctx.now - timedelta(days=90))`.

For instance, if we wanted to calculate the rolling 30 day mean trading volume of MSFT, we can use the expression `market.volume.after(ctx.now - timedelta(days=30)).by_security["MSFT"].as_series().mean()`

### More fleshed out: AAPL volume filter + momentum strategy

Assuming our data period is 1 day, we can implement a momentum/volume filter strategy like below. We keep the universe limited to a single security (AAPL) for simplicity.

```python
def aapl_momentum_with_liquidity(
    universe: Universe,
    current_holdings: Holdings,
    market: MarketView,
    ctx: StrategyContext,
) -> Decision:
    if "AAPL" not in universe:
        return Decision(target=Holdings(), notes={"warn": "AAPL not in universe"})

    aapl_close = market.prices.close.by_security["AAPL"]            # Timeseries[Price, datetime64]
    aapl_tradable = market.tradable.by_security["AAPL"]             # Timeseries[bool, datetime64]
    aapl_volume = (
        market.volume.by_security["AAPL"] if market.volume is not None else None
    )                                                                # Timeseries[Volume, datetime64] | None

    momentum_lookback = 126   # ~6 months
    vol_window = 60           # ~3 months

    # make sure we have enough history
    if len(aapl_close) < momentum_lookback + 1:
        return Decision(target=Holdings(), notes={"info": "warmup - insufficient price history"})

    # momentum: simple ratio of the current price over the price at (lookback) days ago.
    recent_price = aapl_close[-1]
    past_price = aapl_close[-(momentum_lookback + 1)]
    momentum = (recent_price / past_price) - 1.0

    # liquidity filter: average recent volume
    if aapl_volume is not None and len(aapl_volume) >= vol_window:
        avg_vol = aapl_volume[-vol_window:].as_series().mean()
        vol_ok = avg_vol is not None and avg_vol > 0
    else:
        avg_vol = None
        vol_ok = True  # if no volume source, don't block the trade.

    # make sure AAPL is tradable at the decision point.
    tradable_now = bool(aapl_tradable[-1])

    go_long = (momentum > 0.0) and vol_ok and tradable_now
    target = Holdings({"AAPL": 1.0}) if go_long else Holdings()  # 100% AAPL or 100% cash
    # TODO: how to represent no holdings/full cash.

    return Decision(
        target=target,
        notes={
            # TODO: this notes thing seems useful for audit, maybe flesh it out some more, or maybe just a dict is fine.
            "rule": "Long AAPL if 126d momentum > 0 and liquidity/tradability pass",
            "momentum_126d": momentum,
            "avg_vol_60": avg_vol,
            "tradable_now": tradable_now,
        },
    )
```

## Building

- get python 3.13
- run `pip install uv`
- run `uv run python --version` and it will create a venv for you

## Code style

This project is using `ruff` for formatting and linting.

### Formatting

To format the project, run `uv run ruff format`.

### Linting

To lint the project, run `uv run ruff check`.
