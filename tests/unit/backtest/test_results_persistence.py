from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import backtest_lib as btl
from backtest_lib.backtest.results import BacktestResults


def _run_hold_backtest(simple_market) -> BacktestResults:
    initial_capital = 1_000_000
    initial_portfolio = btl.uniform_portfolio(
        simple_market.securities, value=initial_capital
    )

    def strategy():
        return btl.hold()

    return btl.Backtest(
        strategy=strategy,
        market_view=simple_market,
        initial_portfolio=initial_portfolio,
    ).run()


def test_save_and_load_core_bundle(simple_market, tmp_path: Path):
    results = _run_hold_backtest(simple_market)
    bundle_path = tmp_path / "results_bundle"

    saved_path = results.save(bundle_path, metadata={"strategy": "hold"})
    loaded = BacktestResults.load(saved_path)

    assert loaded.initial_capital == pytest.approx(results.initial_capital)
    assert loaded.total_return == pytest.approx(results.total_return)
    assert list(loaded.nav) == pytest.approx(list(results.nav))
    assert list(loaded.portfolio_returns) == pytest.approx(
        list(results.portfolio_returns)
    )
    assert loaded.weights.by_security.to_dataframe(show_periods=True).equals(
        results.weights.by_security.to_dataframe(show_periods=True)
    )


def test_load_from_zip_archive(simple_market, tmp_path: Path):
    results = _run_hold_backtest(simple_market)
    bundle_path = results.save(tmp_path / "bundle", profile="full")

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in bundle_path.rglob("*"):
            if file.is_file():
                archive.write(
                    file,
                    arcname=(
                        Path(bundle_path.name) / file.relative_to(bundle_path)
                    ).as_posix(),
                )

    loaded = BacktestResults.load(zip_path, source="auto")

    assert loaded.total_return == pytest.approx(results.total_return)
    assert list(loaded.nav) == pytest.approx(list(results.nav))
