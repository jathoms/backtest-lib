"""Profiler-friendly cross-sectional momentum backtest.

Examples:
    uv run python examples/profiling_backtest.py
    uv run python -m cProfile -o backtest.prof examples/profiling_backtest.py --repeat 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

import backtest_lib as btl

REPO_ROOT = Path(__file__).resolve().parents[1]
FX_SPOT_CSV = REPO_ROOT / "docs/assets/data/spot_prices.csv"
SP500_CLOSE_PKL = REPO_ROOT / "src/backtest_lib/examples/sp500_close.pkl"

SHORT_WINDOW = 21
MEDIUM_WINDOW = 63
LONG_WINDOW = 126
VOL_WINDOW = 20
VOL_FLOOR = 1e-4


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a profiler-friendly backtest over bundled market data with a "
            "multi-horizon momentum strategy."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "sp500", "fx"),
        default="auto",
        help="Bundled market dataset to use. 'auto' prefers the larger S&P 500 data.",
    )
    parser.add_argument(
        "--schedule",
        default="daily",
        help=(
            "Decision schedule passed into backtest-lib "
            "(for example: daily, weekly, monthly, 2d, or cron)."
        ),
    )
    parser.add_argument(
        "--copies",
        type=positive_int,
        default=1,
        help="Duplicate the security set this many times to scale the workload.",
    )
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=1,
        help="Run the full backtest multiple times in one process for profiling.",
    )
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=25,
        help="Number of names to keep in the momentum basket.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio value.",
    )
    return parser.parse_args()


def load_sp500_frame() -> pl.DataFrame:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The bundled S&P 500 example needs pandas. Run `uv sync --group dev` "
            "or use `--dataset fx`."
        ) from exc

    close_df = pd.read_pickle(SP500_CLOSE_PKL).dropna(axis=1, how="any")
    close_df.index.name = "date"
    return pl.from_pandas(close_df.reset_index())


def load_market_frame(dataset: str) -> tuple[str, pl.DataFrame]:
    if dataset in {"auto", "sp500"} and SP500_CLOSE_PKL.exists():
        return "sp500", load_sp500_frame()

    if dataset == "sp500":
        raise SystemExit(f"Could not find bundled dataset: {SP500_CLOSE_PKL}")

    return "fx", pl.read_csv(FX_SPOT_CSV)


def duplicate_market_frame(frame: pl.DataFrame, copies: int) -> pl.DataFrame:
    if copies == 1:
        return frame

    price_columns = [column for column in frame.columns if column != "date"]
    expressions: list[pl.Expr] = [pl.col("date")]
    for copy_index in range(copies):
        suffix = "" if copy_index == 0 else f"__copy{copy_index + 1}"
        scale = 1.0 + copy_index * 1e-3
        expressions.extend(
            (pl.col(column) * scale).alias(f"{column}{suffix}")
            for column in price_columns
        )
    return frame.select(expressions)


def realized_volatility(prices: list[float]) -> float:
    returns = [
        current_price / previous_price - 1.0
        for previous_price, current_price in zip(prices[:-1], prices[1:], strict=True)
    ]
    mean_return = sum(returns) / len(returns)
    variance = sum((ret - mean_return) ** 2 for ret in returns) / len(returns)
    return variance**0.5


def make_strategy(top_n: int) -> btl.Strategy:
    def momentum_strategy(universe, market, ctx):
        close = market.prices.close
        if len(close.by_period) <= LONG_WINDOW:
            return btl.hold()

        latest = close.by_period[-1]
        short_momentum = latest / close.by_period[-(SHORT_WINDOW + 1)] - 1.0
        medium_momentum = latest / close.by_period[-(MEDIUM_WINDOW + 1)] - 1.0
        long_momentum = latest / close.by_period[-(LONG_WINDOW + 1)] - 1.0
        composite = 0.5 * short_momentum + 0.3 * medium_momentum + 0.2 * long_momentum

        ranked = sorted(
            (
                (security, float(composite[security]))
                for security in universe
                if latest[security] > 0.0 and composite[security] > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        selected = ranked[:top_n]
        if not selected:
            return btl.hold()

        inverse_volatility: dict[str, float] = {}
        for security, _ in selected:
            recent_prices = list(close.by_security[security][-(VOL_WINDOW + 1) :])
            if len(recent_prices) < VOL_WINDOW + 1:
                continue
            vol = max(realized_volatility(recent_prices), VOL_FLOOR)
            inverse_volatility[security] = 1.0 / vol

        if not inverse_volatility:
            return btl.hold()

        total_weight = sum(inverse_volatility.values())
        weights = {
            security: weight / total_weight
            for security, weight in inverse_volatility.items()
        }
        return btl.target_weights(weights, fill_cash=True)

    return momentum_strategy


def build_backtest(
    market: btl.MarketView,
    strategy: btl.Strategy,
    initial_capital: float,
    schedule: str,
) -> btl.Backtest:
    initial_portfolio = btl.uniform_portfolio(market.securities, value=initial_capital)
    return btl.Backtest(
        strategy=strategy,
        market_view=market,
        initial_portfolio=initial_portfolio,
        decision_schedule=schedule,
    )


def format_top_weights(results, top_n: int = 10) -> str:
    final_weights = results.weights.by_period[-1]
    leaders = sorted(
        ((security, float(final_weights[security])) for security in results.securities),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    return ", ".join(f"{security}={weight:.3f}" for security, weight in leaders)


def main() -> None:
    args = parse_args()
    dataset_name, market_frame = load_market_frame(args.dataset)
    market_frame = duplicate_market_frame(market_frame, args.copies)
    market = btl.MarketView(market_frame)
    strategy = make_strategy(args.top_n)

    runtimes: list[float] = []
    results = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        results = build_backtest(
            market=market,
            strategy=strategy,
            initial_capital=args.initial_capital,
            schedule=args.schedule,
        ).run()
        runtimes.append(time.perf_counter() - start)

    assert results is not None

    avg_runtime = sum(runtimes) / len(runtimes)
    sharpe = "n/a" if results.sharpe is None else f"{results.sharpe:.3f}"

    print(
        f"dataset={dataset_name} periods={len(market.periods)} "
        f"securities={len(market.securities)} schedule={args.schedule}"
    )
    print(
        f"repeat={args.repeat} copies={args.copies} "
        f"runtime_s={sum(runtimes):.3f} avg_runtime_s={avg_runtime:.3f}"
    )
    print(
        f"total_return={results.total_return:.3f} "
        f"annualized_return={results.annualized_return:.3f} "
        f"sharpe={sharpe} max_drawdown={results.max_drawdown:.3f} "
        f"avg_turnover={results.avg_turnover:.3f}"
    )
    print(f"ending_nav={results.nav[-1]:,.2f}")
    print(f"top_weights={format_top_weights(results)}")


if __name__ == "__main__":
    main()
