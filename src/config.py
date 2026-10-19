"""Shared settings for the data preparation scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SECTION2_DIR = DATA_DIR / "section2"

TICKERS = [
    "AAPL",
    "SPY",
    "FXE",
    "EWJ",
    "GLD",
    "QQQ",
    "SHV",
    "DBA",
    "USO",
    "XBI",
    "ILF",
    "EPP",
    "FEZ",
]

# The end date is exclusive, so this includes data through 2026-09-30.
START = "2020-01-01"
END = "2026-10-01"

DOWNLOADED_PRICES = DATA_DIR / "prices_downloaded.csv"
PRICES_CLEAN = DATA_DIR / "prices_clean.csv"
DAILY_RETURNS = DATA_DIR / "daily_returns.csv"
CLEANING_REPORT = DATA_DIR / "cleaning_report.csv"
