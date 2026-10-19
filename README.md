# Quantitative Investment Project — Data Preparation

This folder contains the market data and scripts used by the whole group.

## Run the scripts

Run the following commands from the project root, in this order:

```powershell
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py
```

`download_data.py` downloads adjusted daily prices to a temporary file. Then
`prepare_data.py` cleans the prices, calculates simple daily returns, saves the
three final CSV files, and removes the temporary file.

To change the assets or date range, edit `src/config.py`.

## Section 2

Run the Section 2 analysis after preparing the data:

```powershell
python src/section2.py
```

The results are saved under `data/section2/` as CSV files and one PNG figure.
The estimation-input file contains the sample expected returns, covariance
matrix, and individual betas used by the optimizers.

To display the results in the order of the assignment, open
`notebooks/section2_results.ipynb` after running `src/section2.py`.

## Final data files

- `data/prices_clean.csv`: cleaned adjusted prices.
- `data/daily_returns.csv`: daily simple returns used for the analysis.
- `data/cleaning_report.csv`: missing-value summary for each ticker.

The shared analysis file is `data/daily_returns.csv`. It includes AAPL, SPY,
and the 11 ETFs listed in the assignment. The data starts in 2020 to provide
the 60-day warm-up period needed for the Section 6 backtest.
