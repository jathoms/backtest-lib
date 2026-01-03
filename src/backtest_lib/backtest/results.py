from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backtest_lib.market import MarketView, PastView
    from backtest_lib.market.timeseries import Comparable

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults[IndexT: Comparable]:
    """
    Snapshot of a backtest's results, with key statistics pre-computed.

    This is a continuous-weight (no integer position rounding) model.
    """

    periods: Sequence[IndexT] = field(repr=False)
    securities: Sequence[str] = field(repr=False)

    weights: PastView[float, IndexT] = field(repr=False)
    asset_returns: PastView[float, IndexT] = field(repr=False)
    initial_capital: float

    portfolio_returns: list[float]
    nav: list[float]
    drawdowns: list[float]
    gross_exposure: list[float]
    net_exposure: list[float]
    turnover: list[float]

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float | None
    max_drawdown: float
    avg_turnover: float

    market: MarketView[IndexT]
    _backend: type[PastView]

    @staticmethod
    def from_weights_and_returns(
        weights: PastView[float, Any],
        returns: PastView[float, Any],
        market: MarketView[IndexT],
        *,
        initial_capital: float = 1.0,
        periods_per_year: float = 252.0,
        risk_free_annual: float | None = None,
        backend: type[PastView],
    ) -> BacktestResults[Any]:
        """
        Build results from pre-computed per-security simple returns.

        Assumptions:
        - `weights.periods` and `returns.periods` are aligned 1:1.
        - `weights.securities` and `returns.securities` are aligned 1:1.
        - Returns are simple returns over the same period labels as weights.
        """

        periods: Sequence[Any] = weights.periods
        securities: Sequence[str] = weights.securities

        if list(periods) != list(returns.periods):
            raise ValueError("weights and returns must share the same periods")
        if sorted(list(securities)) != sorted(list(returns.securities)):
            raise ValueError("weights and returns must share the same securities")

        n_periods = len(periods)
        n_secs = len(securities)
        if n_periods == 0 or n_secs == 0:
            raise ValueError("Backtest must have at least one period and one security")

        portfolio_returns: list[float] = []
        nav: list[float] = []
        drawdowns: list[float] = []
        gross_exposure: list[float] = []
        net_exposure: list[float] = []
        turnover: list[float] = []

        first_w_vec = weights.by_period[0]
        prev_w_vec = first_w_vec * 0.0

        value = float(initial_capital)
        running_max = value

        for t_idx in range(n_periods):
            w_vec = weights.by_period[t_idx]
            r_vec = returns.by_period[t_idx]

            contrib_vec = w_vec * r_vec

            period_ret = float(contrib_vec.sum())

            gross = float(w_vec.abs().sum())
            net = float(w_vec.sum())

            # turnover: 0.5 * Σ |w_t - w_{t-1}|
            delta_w = w_vec - prev_w_vec
            traded_notional = float(delta_w.abs().sum())
            period_turnover = 0.5 * traded_notional

            prev_w_vec = w_vec

            portfolio_returns.append(period_ret)
            gross_exposure.append(gross)
            net_exposure.append(net)
            turnover.append(period_turnover)

            value *= 1.0 + period_ret
            nav.append(value)

            if value > running_max:
                running_max = value

            dd = value / running_max - 1.0 if running_max > 0 else 0.0
            drawdowns.append(dd)

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs: list[float]) -> float:
            n = len(xs)
            if n < 2:
                return 0.0
            m = _mean(xs)
            var = sum((x - m) * (x - m) for x in xs) / (n - 1)
            return var**0.5

        total_return = nav[-1] / initial_capital - 1.0

        if n_periods > 0:
            annualized_return = (1.0 + total_return) ** (
                periods_per_year / n_periods
            ) - 1.0
        else:
            annualized_return = 0.0

        sigma = _std(portfolio_returns)
        annualized_volatility = sigma * (periods_per_year**0.5)

        max_drawdown = min(drawdowns) if drawdowns else 0.0
        avg_turnover = _mean(turnover) if turnover else 0.0

        if risk_free_annual is None or annualized_volatility == 0.0:
            sharpe = None
        else:
            logger.debug(
                "Calculating sharpe using an annual risk-free-rate of"
                f" {risk_free_annual * 100}% "
            )
            sharpe = (annualized_return - risk_free_annual) / annualized_volatility

        return BacktestResults(
            periods=periods,
            securities=securities,
            weights=weights,
            asset_returns=returns,
            initial_capital=initial_capital,
            portfolio_returns=portfolio_returns,
            nav=nav,
            drawdowns=drawdowns,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            turnover=turnover,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_turnover=avg_turnover,
            market=market,
            _backend=backend,
        )

    @staticmethod
    def from_weights_market_initial_capital(
        weights: PastView[float, Any],
        market: MarketView[Any],
        *,
        initial_capital: float = 1.0,
        periods_per_year: float = 252.0,
        risk_free_annual: float | None = 0.02,
        backend: type[PastView],
    ) -> BacktestResults[Any]:
        """
        Convenience constructor that derives per-security returns from
        `market.prices.close` and then computes all stats.

        Still continuous weight-space (no discrete quantities).
        """

        close_prices = market.prices.close

        if sorted(list(weights.securities)) != sorted(list(close_prices.securities)):
            raise ValueError(
                "weights.securities must match market.prices.close.securities for"
                f" BacktestResults construction (lengths were {len(weights.securities)}"
                f" and {len(close_prices.securities)} respectively)"
            )

        if list(weights.periods) != list(market.periods):
            raise ValueError(
                "weights.periods must match market.periods for BacktestResults "
                "(slice / align before calling, lengths were "
                f"{len(weights.periods)} and {len(market.periods)} respectively)"
            )

        # TODO: big fat polars logic in here, review if this
        # should be handled by the backend itself instead of just
        # using polars.
        #
        # the only downside i see here is having this module be dependent
        # on polars itself (no performance downside, polars pct_change
        # is about as fast as it gets to computing returns).
        close_prices_df = close_prices.by_security.to_dataframe(show_periods=True)

        numeric_cols = [
            name
            for name, dtype in zip(
                close_prices_df.columns, close_prices_df.dtypes, strict=True
            )
            if dtype.is_numeric()
        ]
        asset_returns_df = (
            close_prices_df.lazy()
            .with_columns(pl.col(col).pct_change().alias(col) for col in numeric_cols)
            .with_row_index("idx")
            .with_columns(
                [
                    pl.when(pl.col("idx") == 0).then(0).otherwise(pl.col(c)).alias(c)
                    for c in numeric_cols
                ]
            )
            .drop("idx")
            .select("date", pl.all().exclude("date"))
            .collect()
        )

        asset_returns: PastView = backend.from_dataframe(asset_returns_df)

        results = BacktestResults.from_weights_and_returns(
            weights=weights,
            returns=asset_returns,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
            risk_free_annual=risk_free_annual,
            market=market,
            backend=backend,
        )

        return results

    @cached_property
    def holdings(self) -> PastView[float, IndexT]:
        weights = self.weights.by_security.to_dataframe(lazy=True)
        prices = self.market.prices.close.by_security.to_dataframe(lazy=True)

        joined = weights.join(prices, on="date", suffix="_p")

        weights_schema = weights.collect_schema()
        numeric_cols = [
            name
            for name, dtype in zip(
                weights_schema.names(), weights_schema.dtypes(), strict=True
            )
            if dtype.is_numeric()
        ]

        result = joined.select(
            "date",
            *[(pl.col(c) * pl.col(f"{c}_p")).alias(c) for c in numeric_cols],
        )

        return self._backend.from_dataframe(result.collect())
