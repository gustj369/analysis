# Stock Analysis System

Python script for fetching market data, calculating technical indicators, running a simple golden-cross backtest, and generating Plotly dashboard HTML files.

## Features

- Fetches global assets with `yfinance`
- Fetches Korean stocks with `FinanceDataReader`
- Calculates MA, RSI, MACD, return, volatility, Sharpe ratio, and MDD
- Compares a golden-cross strategy with buy-and-hold
- Generates interactive Plotly dashboards

## Project Files

- `stock_analysis.py`: main analysis script and CLI entry point
- `requirements.txt`: Python dependencies
- `AGENTS.md`: coding-agent guide for this project
- `*_dashboard.html`: generated dashboard output files, ignored by Git

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

Run with the default settings:

```powershell
python stock_analysis.py
```

Run a specific ticker:

```powershell
python stock_analysis.py TSLA --years 3
```

Run without generating a chart:

```powershell
python stock_analysis.py BTC-USD --years 1 --no-chart
```

Use explicit dates:

```powershell
python stock_analysis.py AAPL --start 2022-01-01 --end 2024-01-01
```

## Tests

Run the network-free unit tests:

```powershell
python -m unittest -v
```

By default, tests do not fetch live market data. To include the optional external data smoke test:

```powershell
$env:RUN_NETWORK_TESTS = "1"
python -m unittest -v
Remove-Item Env:\RUN_NETWORK_TESTS
```

Plotly dashboard structure tests are skipped automatically if Plotly is not installed.
Install dependencies first if you want the Plotly test to run instead of skip:

```powershell
pip install -r requirements.txt
python -m unittest -v
```

## GitHub Upload

If an automated environment cannot write to `.git` or cannot reach GitHub, run the final Git commands directly from your local PowerShell:

```powershell
git add .gitignore README.md requirements.txt stock_analysis.py test_stock_analysis.py
git commit -m "Add README and expanded tests"
git push -u origin main
```

## Notes

- Market data depends on external data providers and may change over time.
- Generated dashboard HTML files are intentionally ignored by Git.
