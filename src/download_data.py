"""Download the prices used in the project."""

import pandas as pd
import yfinance as yf

from config import DATA_DIR, DOWNLOADED_PRICES, END, START, TICKERS


def download_prices():
    DATA_DIR.mkdir(exist_ok=True)

    downloaded = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("No price data was downloaded. Check the network connection.")

    prices = downloaded["Close"]
    missing_tickers = set(TICKERS) - set(prices.columns)
    if missing_tickers:
        raise RuntimeError(f"Missing tickers in the download: {sorted(missing_tickers)}")

    prices = prices.reindex(columns=TICKERS)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices.to_csv(DOWNLOADED_PRICES, index_label="Date")
    print("Downloaded prices. Run src/prepare_data.py next.")


if __name__ == "__main__":
    download_prices()
