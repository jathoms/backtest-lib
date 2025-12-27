import datetime as dt
import pickle as pkl
import time
from importlib.resources import files

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

import backtest_lib.examples
from backtest_lib.market import MarketView, PastUniversePrices, PastView
from backtest_lib.market.polars_impl import PolarsPastView
from backtest_lib.market.polars_impl._helpers import Array1DDTView


def fetch_history(tickers, start, end=None, interval="1d"):
    tickers = [t.replace(".", "-") for t in tickers]
    all_data = []
    chunk_size = 50

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        print(f"Fetching {batch[0]}...{batch[-1]}")

        df = yf.download(
            tickers=batch,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        all_data.append(df)

        time.sleep(1)  # avoid being throttled

    return all_data


def fetch_history_one(tickers, start, end=None, interval="1d"):
    tickers = [t.replace(".", "-") for t in tickers]
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )
    return df


def get_sp500_market_view(
    start: dt.datetime,
    end: dt.datetime | None = dt.datetime.now(),
) -> MarketView[np.datetime64]:
    changes = pkl.load(
        files(backtest_lib.examples).joinpath("sp500_changes.pkl").open("rb"),
    )

    current = pkl.load(
        files(backtest_lib.examples).joinpath("sp500_const.pkl").open("rb"),
    )

    dates = pd.date_range(start, end, freq="B").values

    added = changes[["Effective Date", "Added"]]
    added.columns = added.columns.droplevel(0)
    added = added.loc[:, ["Effective Date", "Ticker"]]
    added.loc[:, "Effective Date"] = pd.to_datetime(added["Effective Date"])
    added.dropna(inplace=True)

    removed = changes[["Effective Date", "Removed"]]
    removed.columns = removed.columns.droplevel(0)
    removed = removed.loc[:, ["Effective Date", "Ticker"]]
    removed.loc[:, "Effective Date"] = pd.to_datetime(removed["Effective Date"])
    removed.dropna(inplace=True)

    current_tickers = [
        line.rstrip("\n")
        for line in (
            files(backtest_lib.examples)
            .joinpath("sp500_constituents.txt")
            .open("r")
            .readlines()
        )
    ]
    all_historical_tickers = (
        set(added["Ticker"]).union(set(removed["Ticker"])).union(current_tickers)
    )

    tradeable_df = pd.DataFrame(
        {
            "date": dates,
            **{ticker: [True] * len(dates) for ticker in all_historical_tickers},
        },
    )

    for ticker in all_historical_tickers:
        date_first_added_df = current.query("Symbol == @ticker")["Date added"]
        date_first_added = (
            date_first_added_df.iat[0] if not date_first_added_df.empty else None
        )
        dates_added: pd.Series = added.query("Ticker == @ticker")["Effective Date"]
        dates_removed: pd.Series = removed.query("Ticker == @ticker")["Effective Date"]

        poi_added = [(date, "a") for date in dates_added]
        poi_rem = [(date, "r") for date in dates_removed]

        poi = sorted((poi_added + poi_rem), key=lambda tup: tup[0])
        first_addition = True

        if date_first_added is not None:
            tradeable_df.loc[tradeable_df["date"] < date_first_added, ticker] = False

        for date, action in poi:
            if action == "a":
                if first_addition:
                    tradeable_df.loc[tradeable_df["date"] < date, ticker] = False
                    first_addition = False
                tradeable_df.loc[tradeable_df["date"] >= date, ticker] = True
            if action == "r":
                tradeable_df.loc[tradeable_df["date"] >= date, ticker] = False

    tradable_view = PolarsPastView.from_dataframe(tradeable_df)

    tickers_to_fetch = [
        ticker
        for ticker in all_historical_tickers
        if any(x for x in tradable_view.after(start.isoformat()).by_security[ticker])
    ]
    history = fetch_history_one(tickers_to_fetch, start, end)
    # history = pkl.load(
    #     files(backtest_lib.examples).joinpath("sp500_more_tk.pkl").open("rb")
    # )
    close_df = (
        history.xs("Close", axis=1, level=1)
        .dropna(axis=1, how="any")
        .reindex(dates)
        .ffill()
    )
    close_pl = pl.from_pandas(close_df)

    close_prices_df = close_pl.with_columns(pl.Series("date", dates)).fill_null(0)
    while all(x == 0 for x in close_prices_df.row(0)):
        close_prices_df = close_prices_df.slice(1)
        tradable_view = tradable_view.by_period[1:]
    close_price_past_view: PastView[float, np.datetime64] = (
        PolarsPastView.from_dataframe(close_prices_df)
    )
    tradable_view = tradable_view.by_security[close_price_past_view.securities]

    return MarketView(
        prices=PastUniversePrices(close=close_price_past_view),
        tradable=tradable_view,
    )
