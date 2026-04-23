from __future__ import annotations

import io
import json
import logging
import shutil
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from os import PathLike, fspath
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from backtest_lib.market._backends import (
    _get_pastview_type_from_backend,
    _get_timeseries_type_from_backend,
)
from backtest_lib.market.timeseries import Timeseries

if TYPE_CHECKING:
    from backtest_lib.market import MarketView, PastView
    from backtest_lib.market.timeseries import Comparable

import polars as pl

logger = logging.getLogger(__name__)


type PathInput = str | PathLike[str]
type SaveProfile = Literal["core", "analysis", "full"]
type IfExistsPolicy = Literal["error", "overwrite"]
type LoadValidationMode = Literal["strict", "warn", "none"]
type LoadSource = Literal["reconstruct", "auto"]
type MetadataValue = str | int | float | bool | None

_BUNDLE_SCHEMA_VERSION = 1
_BUNDLE_TYPE = "backtest_results"
_MANIFEST_FILENAME = "manifest.json"
_SUMMARY_FILENAME = "summary.json"
_RUN_CONFIG_FILENAME = "run_config.json"
_WEIGHTS_FILENAME = "weights.parquet"
_PRICES_CLOSE_FILENAME = "prices_close.parquet"
_TIMESERIES_FILENAME = "timeseries.parquet"
_ASSET_RETURNS_FILENAME = "asset_returns.parquet"
_VALUES_HELD_FILENAME = "values_held.parquet"
_QUANTITIES_HELD_FILENAME = "quantities_held.parquet"


@dataclass
class BacktestResults[IndexT: Comparable]:
    """Snapshot of a backtest's results, with key statistics pre-computed."""

    periods: Sequence[IndexT] = field(repr=False)
    securities: Sequence[str] = field(repr=False)

    weights: PastView[float, IndexT] = field(repr=False)
    asset_returns: PastView[float, IndexT] = field(repr=False)
    initial_capital: float

    portfolio_returns: Timeseries[float, IndexT]
    nav: Timeseries[float, IndexT]
    drawdowns: Timeseries[float, IndexT]
    gross_exposure: Timeseries[float, IndexT]
    net_exposure: Timeseries[float, IndexT]
    turnover: Timeseries[float, IndexT]

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float | None
    max_drawdown: float
    avg_turnover: float

    market: MarketView[IndexT]
    _backend: str
    _backend_pastview_type: type[PastView]
    _backend_timeseries_type: type[Timeseries]
    _periods_per_year: float = field(repr=False, default=252.0)
    _risk_free_annual: float | None = field(repr=False, default=None)

    @staticmethod
    def from_weights_and_returns(
        weights: PastView[float, Any],
        returns: PastView[float, Any],
        market: MarketView[IndexT],
        *,
        initial_capital: float = 1.0,
        periods_per_year: float = 252.0,
        risk_free_annual: float | None = None,
        backend: str = "polars",
    ) -> BacktestResults[Any]:
        """Build results from pre-computed per-security simple returns.

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

        timeseries_type = _get_timeseries_type_from_backend(backend)

        return BacktestResults(
            periods=periods,
            securities=securities,
            weights=weights,
            asset_returns=returns,
            initial_capital=initial_capital,
            portfolio_returns=timeseries_type.from_vectors(portfolio_returns, periods),
            nav=timeseries_type.from_vectors(nav, periods),
            drawdowns=timeseries_type.from_vectors(drawdowns, periods),
            gross_exposure=timeseries_type.from_vectors(gross_exposure, periods),
            net_exposure=timeseries_type.from_vectors(net_exposure, periods),
            turnover=timeseries_type.from_vectors(turnover, periods),
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_turnover=avg_turnover,
            market=market,
            _backend=backend,
            _backend_pastview_type=_get_pastview_type_from_backend(backend),
            _backend_timeseries_type=timeseries_type,
            _periods_per_year=periods_per_year,
            _risk_free_annual=risk_free_annual,
        )

    @staticmethod
    def from_weights_market_initial_capital(
        weights: PastView[float, Any],
        market: MarketView[Any],
        initial_capital: float,
        *,
        periods_per_year: float = 252.0,
        risk_free_annual: float | None = 0.02,
        backend: str,
    ) -> BacktestResults[Any]:
        """Convenience constructor that derives per-security returns from
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

        backend_pastview_type = _get_pastview_type_from_backend(backend)

        asset_returns: PastView = backend_pastview_type.from_dataframe(asset_returns_df)

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

    def save(
        self,
        path: PathInput,
        *,
        profile: SaveProfile = "core",
        if_exists: IfExistsPolicy = "error",
        compression: Literal["zstd", "snappy", "gzip"] | None = "zstd",
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> Path:
        """Persist results as a directory bundle.

        The default ``core`` profile stores a lightweight, reconstructable subset:
        summary metrics, run configuration, weights, and close prices.
        """
        if profile not in {"core", "analysis", "full"}:
            raise ValueError(
                f"Unknown profile: {profile!r}. Valid profiles are"
                " 'core', 'analysis', and 'full'."
            )
        if if_exists not in {"error", "overwrite"}:
            raise ValueError(
                f"Unknown if_exists mode: {if_exists!r}. Valid values are"
                " 'error' and 'overwrite'."
            )

        bundle_path = Path(fspath(path))
        if bundle_path.exists():
            if if_exists == "error":
                raise FileExistsError(f"Path already exists: {bundle_path}")
            if bundle_path.is_dir():
                shutil.rmtree(bundle_path)
            else:
                bundle_path.unlink()
        bundle_path.mkdir(parents=True, exist_ok=False)

        parquet_compression = compression or "uncompressed"
        metadata_payload = _normalise_metadata(metadata)

        files: dict[str, str] = {
            "summary": _SUMMARY_FILENAME,
            "run_config": _RUN_CONFIG_FILENAME,
            "weights": _WEIGHTS_FILENAME,
            "prices_close": _PRICES_CLOSE_FILENAME,
        }

        summary_payload = {
            "initial_capital": self.initial_capital,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "avg_turnover": self.avg_turnover,
            "n_periods": len(self.periods),
            "n_securities": len(self.securities),
        }
        _write_json(bundle_path / _SUMMARY_FILENAME, summary_payload)

        run_config_payload = {
            "initial_capital": self.initial_capital,
            "periods_per_year": self._periods_per_year,
            "risk_free_annual": self._risk_free_annual,
        }
        _write_json(bundle_path / _RUN_CONFIG_FILENAME, run_config_payload)

        weights_df = self.weights.by_security.to_dataframe(show_periods=True)
        weights_df.write_parquet(
            bundle_path / _WEIGHTS_FILENAME,
            compression=parquet_compression,
        )

        prices_close_df = self.market.prices.close.by_security.to_dataframe(
            show_periods=True
        )
        prices_close_df.write_parquet(
            bundle_path / _PRICES_CLOSE_FILENAME,
            compression=parquet_compression,
        )

        if profile in {"analysis", "full"}:
            period_series = pl.Series(
                "date",
                np.asarray(self.periods, dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            )
            timeseries_df = pl.DataFrame(
                {
                    "date": period_series,
                    "portfolio_returns": self.portfolio_returns.to_series(),
                    "nav": self.nav.to_series(),
                    "drawdowns": self.drawdowns.to_series(),
                    "gross_exposure": self.gross_exposure.to_series(),
                    "net_exposure": self.net_exposure.to_series(),
                    "turnover": self.turnover.to_series(),
                }
            )
            timeseries_df.write_parquet(
                bundle_path / _TIMESERIES_FILENAME,
                compression=parquet_compression,
            )
            files["timeseries"] = _TIMESERIES_FILENAME

        if profile == "full":
            asset_returns_df = self.asset_returns.by_security.to_dataframe(
                show_periods=True
            )
            asset_returns_df.write_parquet(
                bundle_path / _ASSET_RETURNS_FILENAME,
                compression=parquet_compression,
            )
            files["asset_returns"] = _ASSET_RETURNS_FILENAME

            values_held_df = self.values_held.by_security.to_dataframe(
                show_periods=True
            )
            values_held_df.write_parquet(
                bundle_path / _VALUES_HELD_FILENAME,
                compression=parquet_compression,
            )
            files["values_held"] = _VALUES_HELD_FILENAME

            quantities_held_df = self.quantities_held.by_security.to_dataframe(
                show_periods=True
            )
            quantities_held_df.write_parquet(
                bundle_path / _QUANTITIES_HELD_FILENAME,
                compression=parquet_compression,
            )
            files["quantities_held"] = _QUANTITIES_HELD_FILENAME

        manifest_payload = {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "bundle_type": _BUNDLE_TYPE,
            "profile": profile,
            "backend": self._backend,
            "created_at": datetime.now(UTC).isoformat(),
            "files": files,
            "metadata": metadata_payload,
        }
        _write_json(bundle_path / _MANIFEST_FILENAME, manifest_payload)

        return bundle_path

    @classmethod
    def load(
        cls,
        path: PathInput,
        *,
        validate: LoadValidationMode = "strict",
        source: LoadSource = "reconstruct",
    ) -> BacktestResults[Any]:
        """Load results from a bundle directory or zip archive.

        This method always discovers the bundle by locating exactly one
        ``manifest.json``.
        """
        bundle_path = Path(fspath(path))
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")

        if bundle_path.is_dir():
            manifest_paths = sorted(bundle_path.rglob(_MANIFEST_FILENAME))
            if not manifest_paths:
                raise ValueError(f"No '{_MANIFEST_FILENAME}' found under {bundle_path}")
            if len(manifest_paths) > 1:
                raise ValueError(
                    "Expected exactly one manifest in bundle directory, found"
                    f" {len(manifest_paths)}: {manifest_paths}"
                )
            manifest_path = manifest_paths[0]
            root = manifest_path.parent
            manifest = _read_json(manifest_path)

            def read_json(rel_path: str) -> dict[str, Any]:
                return _read_json(root / rel_path)

            def read_parquet(rel_path: str) -> pl.DataFrame:
                return pl.read_parquet(root / rel_path)

            def exists(rel_path: str) -> bool:
                return (root / rel_path).exists()

            return cls._load_from_bundle(
                manifest=manifest,
                read_json=read_json,
                read_parquet=read_parquet,
                exists=exists,
                validate=validate,
                source=source,
            )

        if not zipfile.is_zipfile(bundle_path):
            raise ValueError(
                f"Expected a bundle directory or zip archive, got: {bundle_path}"
            )

        with zipfile.ZipFile(bundle_path) as zf:
            members = zf.namelist()
            manifest_members = [
                member
                for member in members
                if PurePosixPath(member).name == _MANIFEST_FILENAME
            ]
            if not manifest_members:
                raise ValueError(
                    f"No '{_MANIFEST_FILENAME}' found in zip archive {bundle_path}"
                )
            if len(manifest_members) > 1:
                raise ValueError(
                    "Expected exactly one manifest in zip archive, found"
                    f" {len(manifest_members)}: {manifest_members}"
                )

            manifest_member = manifest_members[0]
            manifest_dir = PurePosixPath(manifest_member).parent
            members_set = set(members)
            manifest = json.loads(zf.read(manifest_member).decode("utf-8"))

            def resolve_member(rel_path: str) -> str:
                rel = PurePosixPath(rel_path)
                if str(manifest_dir) in {"", "."}:
                    return rel.as_posix()
                return (manifest_dir / rel).as_posix()

            def read_json(rel_path: str) -> dict[str, Any]:
                member = resolve_member(rel_path)
                return json.loads(zf.read(member).decode("utf-8"))

            def read_parquet(rel_path: str) -> pl.DataFrame:
                member = resolve_member(rel_path)
                return pl.read_parquet(io.BytesIO(zf.read(member)))

            def exists(rel_path: str) -> bool:
                return resolve_member(rel_path) in members_set

            return cls._load_from_bundle(
                manifest=manifest,
                read_json=read_json,
                read_parquet=read_parquet,
                exists=exists,
                validate=validate,
                source=source,
            )

    @classmethod
    def _load_from_bundle(
        cls,
        *,
        manifest: Mapping[str, Any],
        read_json: Callable[[str], dict[str, Any]],
        read_parquet: Callable[[str], pl.DataFrame],
        exists: Callable[[str], bool],
        validate: LoadValidationMode,
        source: LoadSource,
    ) -> BacktestResults[Any]:
        if validate not in {"strict", "warn", "none"}:
            raise ValueError(
                f"Unknown validate mode: {validate!r}. Valid values are"
                " 'strict', 'warn', and 'none'."
            )
        if source not in {"reconstruct", "auto"}:
            raise ValueError(
                f"Unknown source mode: {source!r}. Valid values are"
                " 'reconstruct' and 'auto'."
            )

        schema_version = manifest.get("schema_version")
        if schema_version != _BUNDLE_SCHEMA_VERSION:
            _handle_validation_issue(
                "Unexpected bundle schema version. Expected"
                f" {_BUNDLE_SCHEMA_VERSION}, got {schema_version!r}.",
                validate,
            )

        bundle_type = manifest.get("bundle_type")
        if bundle_type != _BUNDLE_TYPE:
            _handle_validation_issue(
                f"Unexpected bundle_type. Expected {_BUNDLE_TYPE!r},"
                f" got {bundle_type!r}.",
                validate,
            )

        files_obj = manifest.get("files")
        if not isinstance(files_obj, Mapping):
            raise ValueError("Invalid bundle manifest: 'files' must be a mapping.")

        files = {
            str(key): value
            for key, value in files_obj.items()
            if isinstance(key, str) and isinstance(value, str)
        }

        required_keys = ("run_config", "weights", "prices_close")
        missing_keys = [key for key in required_keys if key not in files]
        if missing_keys:
            raise ValueError(
                f"Invalid bundle manifest. Missing required file keys: {missing_keys}"
            )

        missing_files = [files[key] for key in required_keys if not exists(files[key])]
        if missing_files:
            raise FileNotFoundError(
                "Bundle manifest references missing files: " + ", ".join(missing_files)
            )

        backend = manifest.get("backend", "polars")
        if not isinstance(backend, str):
            raise ValueError("Invalid bundle manifest: 'backend' must be a string.")

        run_config = read_json(files["run_config"])
        if not isinstance(run_config, Mapping):
            raise ValueError("Invalid run config: expected a JSON object.")

        try:
            initial_capital = float(run_config["initial_capital"])
        except KeyError as e:
            raise ValueError("Invalid run config: missing 'initial_capital'.") from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Invalid run config: 'initial_capital' must be numeric."
            ) from e

        try:
            periods_per_year = float(run_config.get("periods_per_year", 252.0))
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Invalid run config: 'periods_per_year' must be numeric."
            ) from e

        risk_free_raw = run_config.get("risk_free_annual", None)
        if risk_free_raw is None:
            risk_free_annual = None
        else:
            try:
                risk_free_annual = float(risk_free_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Invalid run config: 'risk_free_annual' must be numeric or null."
                ) from e

        weights_df = read_parquet(files["weights"])
        prices_close_df = read_parquet(files["prices_close"])

        backend_pastview_type = _get_pastview_type_from_backend(backend)
        weights = backend_pastview_type.from_dataframe(weights_df)
        from backtest_lib.market import MarketView

        market = MarketView(prices=prices_close_df, backend=backend)

        asset_returns_file = files.get("asset_returns")
        if (
            source == "auto"
            and asset_returns_file is not None
            and exists(asset_returns_file)
        ):
            asset_returns_df = read_parquet(asset_returns_file)
            asset_returns = backend_pastview_type.from_dataframe(asset_returns_df)
            return cls.from_weights_and_returns(
                weights=weights,
                returns=asset_returns,
                market=market,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
                risk_free_annual=risk_free_annual,
                backend=backend,
            )

        return cls.from_weights_market_initial_capital(
            weights=weights,
            market=market,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
            risk_free_annual=risk_free_annual,
            backend=backend,
        )

    @cached_property
    def quantities_held(self) -> PastView[float, IndexT]:
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

        qtys = joined.select(
            "date",
            *[
                (pl.col(c) * self.nav.to_series() / pl.col(f"{c}_p")).alias(c)
                for c in numeric_cols
            ],
        )

        return self._backend_pastview_type.from_dataframe(qtys.collect())

    @cached_property
    def values_held(self) -> PastView[float, IndexT]:
        weights = self.weights.by_security.to_dataframe(lazy=True)

        weights_schema = weights.collect_schema()
        numeric_cols = [
            name
            for name, dtype in zip(
                weights_schema.names(), weights_schema.dtypes(), strict=True
            )
            if dtype.is_numeric()
        ]

        values = weights.select(
            "date",
            *[(pl.col(c) * self.nav.to_series()).alias(c) for c in numeric_cols],
        )

        return self._backend_pastview_type.from_dataframe(values.collect())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _handle_validation_issue(message: str, validate: LoadValidationMode) -> None:
    if validate == "strict":
        raise ValueError(message)
    if validate == "warn":
        warnings.warn(message, stacklevel=3)


def _normalise_metadata(
    metadata: Mapping[str, MetadataValue] | None,
) -> dict[str, MetadataValue]:
    if metadata is None:
        return {}

    normalised: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError(
                f"metadata keys must be strings. Got key {key!r} of type {type(key)}."
            )
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError(
                "metadata values must be scalar JSON-compatible values "
                "(str, int, float, bool, or None). "
                f"Got metadata[{key!r}]={value!r}."
            )
        normalised[key] = value
    return normalised
