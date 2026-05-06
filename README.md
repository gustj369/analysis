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

The analysis also prints a simple benchmark comparison.
It shows total return, annualized return, MDD, excess return, and latest rolling correlation versus each benchmark for the same available date range.
Generated Plotly dashboards also include a compact benchmark excess-return and rolling-correlation summary near the top.
The dashboard includes cumulative-return and/or excess-return comparison lines below the MACD panel.
The default benchmark panel shows cumulative-return lines only, keeping excess-return lines available through `--benchmark-display all` or `--benchmark-display excess`.
Use `--chart-mode full` when you want all benchmark lines, signal markers, and benchmark legend entries visible together.

Default benchmarks are selected by asset type:

- Global stocks: `SPY`, `QQQ`
- Korean stocks: `EWY`, `SPY`
- Crypto assets: `BTC-USD`, `ETH-USD` with the analyzed ticker removed when duplicated

You can also choose a preset explicitly with `--benchmark-preset auto|us|korea|crypto|off`.

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

Korean stock runs print a small data sanity check for latest close, adjusted-close ratio, close-price range, and yfinance `.KS/.KQ` reference price when available.

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

Choose a benchmark preset:

```powershell
python stock_analysis.py BTC-USD --years 1 --benchmark-preset crypto
python stock_analysis.py AAPL --years 1 --benchmark-preset off
```

Choose a chart mode:

```powershell
python stock_analysis.py AAPL --years 1 --chart-mode clean
python stock_analysis.py AAPL --years 1 --chart-mode full
```

Hide excess-return lines when the benchmark panel gets crowded:

```powershell
python stock_analysis.py AAPL --years 1 --no-excess-line
```

Hide auxiliary signal markers from the legend:

```powershell
python stock_analysis.py AAPL --years 1 --hide-marker-legend
```

Hide auxiliary signal markers from the chart:

```powershell
python stock_analysis.py AAPL --years 1 --hide-signal-markers
python stock_analysis.py AAPL --years 1 --show-signal-markers
```

Show benchmark lines in the legend:

```powershell
python stock_analysis.py AAPL --years 1 --chart-mode full --show-benchmark-legend
```

Debug benchmark source data:

```powershell
python stock_analysis.py AAPL --years 1 --no-chart --debug-benchmarks
```

Use this when benchmark rows look suspiciously identical. It prints each benchmark's first close, last close, and row count.

Choose the benchmark panel line mode or rolling-correlation window:

```powershell
python stock_analysis.py AAPL --years 1 --benchmark-display cumulative
python stock_analysis.py AAPL --years 1 --benchmark-display all
python stock_analysis.py AAPL --years 1 --benchmark-display excess --corr-window 30
```

Programmatic runs return the Korean-data checks as dictionaries:

```python
from stock_analysis import run_analysis

result = run_analysis("005930", show_chart=False)
print(result["data_quality"])
print(result["external_price_check"])
print(result["strategy_summary"])
```

`data_quality["warnings"]` may include values such as `adjusted_close_diff`, `wide_close_range`, `latest_close_outlier`, or `large_daily_move`.
For Korean stocks, treat these as a prompt to compare the latest close against another data source before interpreting returns.
Generated dashboards also surface these warning codes in the top summary.

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

For a visual dashboard check on a local machine with Plotly installed:

```powershell
python stock_analysis.py AAPL --years 1
python stock_analysis.py AAPL --years 1 --chart-mode full
python stock_analysis.py AAPL --years 1 --hide-marker-legend
python stock_analysis.py AAPL --years 1 --hide-signal-markers
```

Open the generated HTML and capture a screenshot. Check that the top summary separates current state, performance, benchmark, strategy, and any data-warning lines; the candlestick chart is not blank; benchmark lines appear in the taller bottom panel; and `full` mode shows grouped legend entries for price, indicators, and benchmarks.
For Korean stocks with data warnings, confirm that the top summary shows the warning line in a distinct accent color.

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
