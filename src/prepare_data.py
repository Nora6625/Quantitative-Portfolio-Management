"""Clean prices and create the three data files shared by the group."""

import pandas as pd

from config import (
    CLEANING_REPORT,
    DAILY_RETURNS,
    DOWNLOADED_PRICES,
    PRICES_CLEAN,
    TICKERS,
)


def prepare_data():
    raw_prices = pd.read_csv(DOWNLOADED_PRICES, index_col="Date", parse_dates=["Date"])
    raw_prices = raw_prices[TICKERS].sort_index()

    if raw_prices.empty:
        raise RuntimeError("The downloaded price file is empty. Run download_data.py again.")

    report = pd.DataFrame(
        {
            "missing_count_before_cleaning": raw_prices.isna().sum(),
            "missing_fraction_before_cleaning": raw_prices.isna().mean(),
        }
    )
    report.index.name = "Ticker"

    prices_clean = raw_prices.dropna()
    daily_returns = prices_clean.pct_change().dropna()

    if prices_clean.empty or daily_returns.empty:
        raise RuntimeError("No complete price and return rows remain after cleaning.")

    prices_clean.to_csv(PRICES_CLEAN, index_label="Date")
    daily_returns.to_csv(DAILY_RETURNS, index_label="Date")
    report.to_csv(CLEANING_REPORT)

    # The downloaded table is only an intermediate file. The three files
    # above are the complete data package shared with the group.
    DOWNLOADED_PRICES.unlink()
    print("Created prices_clean.csv, daily_returns.csv, and cleaning_report.csv.")
    print("\nPreview of daily_returns.csv:")
    print(daily_returns.head())
    print(f"\nShape: {daily_returns.shape}")


if __name__ == "__main__":
    prepare_data()
