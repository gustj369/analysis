# Stock Analysis System

Python script for fetching market data, calculating technical indicators, running a simple golden-cross backtest, and generating Plotly dashboard HTML files.

## Features

- Fetches global assets with `yfinance`
- Fetches Korean stocks with `FinanceDataReader`
- Calculates MA, RSI, MACD, return, volatility, Sharpe ratio, and MDD
- Compares a golden-cross strategy with buy-and-hold
- Prints a current-state summary for trend, RSI, MACD, and signal score
- Generates interactive Plotly dashboards

## Project Files

- `stock_analysis.py`: main analysis script and CLI entry point
- `requirements.txt`: Python dependencies
- `AGENTS.md`: coding-agent guide for this project
- `*_dashboard.html`: generated dashboard output files, ignored by Git

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Usage

### Latest Analysis

Analyze the latest available one-year data and generate a dashboard:

```powershell
python stock_analysis.py AAPL --years 1
```

Analyze the latest available one-year data without generating a chart:

```powershell
python stock_analysis.py AAPL --years 1 --no-chart
```

The console output includes a current-state summary such as trend direction, price position versus moving averages, RSI state, MACD momentum, and a 0-5 signal score.
The same summary is also shown near the top of generated Plotly dashboards.

Signal score is a simple 5-point checklist:

- Close is above MA20
- Close is above MA60
- MA20 is above MA60
- MACD is above the MACD signal line
- RSI is between 40 and 70

This score is a quick technical snapshot, not a buy/sell recommendation.

The analysis also prints a simple benchmark comparison against `SPY` and `QQQ`.
It shows total return, annualized return, MDD, and the analyzed ticker's excess return versus each benchmark for the same available date range.
Generated Plotly dashboards also include a compact benchmark excess-return summary near the top.
The dashboard includes a cumulative-return comparison line chart below the MACD panel.

Default benchmarks are selected by asset type:

- Global stocks: `SPY`, `QQQ`
- Korean stocks: `EWY`, `SPY`
- Crypto assets: `BTC-USD`, `ETH-USD` with the analyzed ticker removed when duplicated

### Fixed Historical Period

Analyze a specific historical period:

```powershell
python stock_analysis.py AAPL --start 2024-01-01 --end 2024-03-15
```

Fixed historical periods are useful for stable tests and repeatable comparisons. Latest analysis should use `--years`.

### Other Examples

Run with the default settings:

```powershell
python stock_analysis.py
```

Run a specific ticker:

```powershell
python stock_analysis.py TSLA --years 3
```

Run a Korean stock:

```powershell
python stock_analysis.py 005930 --years 2
```

Run a crypto asset:

```powershell
python stock_analysis.py BTC-USD --years 1
```

Run without generating a chart:

```powershell
python stock_analysis.py BTC-USD --years 1 --no-chart
```

Use explicit dates:

```powershell
python stock_analysis.py AAPL --start 2022-01-01 --end 2024-01-01
```

Overwrite the dashboard HTML instead of using a timestamped filename:

```powershell
python stock_analysis.py AAPL --overwrite
```

Choose custom benchmarks:

```powershell
python stock_analysis.py AAPL --years 1 --benchmarks SPY,QQQ,VTI
```

## Tests

Run the network-free unit tests:

```powershell
python -m unittest -v
```

By default, tests do not fetch live market data. This keeps everyday verification fast and deterministic.
To include the optional external data smoke test:

```powershell
$env:RUN_NETWORK_TESTS = "1"
python -m unittest -v
Remove-Item Env:\RUN_NETWORK_TESTS
```

Plotly dashboard structure tests are skipped automatically if Plotly is not installed.
Install dependencies first if you want the Plotly test to run instead of skip:

```powershell
python -m pip install -r requirements.txt
python -m unittest -v
```

## GitHub Upload

If an automated environment cannot write to `.git` or cannot reach GitHub, run the final Git commands directly from your local PowerShell:

```powershell
git add .gitignore README.md requirements.txt stock_analysis.py test_stock_analysis.py
git commit -m "Add backtest tests and README examples"
git push -u origin main
```

## Notes

- Market data depends on external data providers and may change over time.
- Generated dashboard HTML files are intentionally ignored by Git.
- Benchmark comparison uses `SPY` and `QQQ` as broad U.S. market references. For Korean stocks or crypto assets, interpret this as a rough external comparison rather than a perfect peer benchmark.
