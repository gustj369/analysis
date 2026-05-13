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

### Quick Checks

Run the default analysis:

```powershell
python stock_analysis.py
```

Analyze the latest available one-year data without generating a chart:

```powershell
python stock_analysis.py AAPL --years 1 --no-chart
```

Run common asset types:

```powershell
python stock_analysis.py TSLA --years 3 --no-chart
python stock_analysis.py 005930 --years 2 --no-chart
python stock_analysis.py BTC-USD --years 1 --no-chart
```

Korean stock runs print a small data sanity check for latest close, adjusted-close ratio, close-price range, and yfinance `.KS/.KQ` reference price when available.

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
Use `--chart-mode full` when you want all benchmark lines and signal markers visible together. Auxiliary marker and benchmark legend entries stay hidden by default to keep the chart readable.

Default benchmarks are selected by asset type:

- Global stocks: `SPY`, `QQQ`
- Korean stocks: `EWY`, `SPY`
- Crypto assets: `BTC-USD`, `ETH-USD` with the analyzed ticker removed when duplicated

You can also choose a preset explicitly with `--benchmark-preset auto|us|korea|crypto|off`.

### Chart Generation

Generate a dashboard with the default clean chart mode:

```powershell
python stock_analysis.py AAPL --years 1
```

Choose a chart mode:

```powershell
python stock_analysis.py AAPL --years 1 --chart-mode clean
python stock_analysis.py AAPL --years 1 --chart-mode full
```

Overwrite the dashboard HTML instead of using a timestamped filename:

```powershell
python stock_analysis.py AAPL --overwrite
```

Save dashboard files to a specific folder:

```powershell
python stock_analysis.py AAPL --years 1 --output-dir reports
```

Auxiliary signal marker legend entries are hidden by default. Show them when needed:

```powershell
python stock_analysis.py AAPL --years 1 --chart-mode full --show-marker-legend
```

Hide or show auxiliary signal markers:

```powershell
python stock_analysis.py AAPL --years 1 --hide-signal-markers
python stock_analysis.py AAPL --years 1 --show-signal-markers
```

### Benchmark Options

```powershell
python stock_analysis.py AAPL --years 1 --benchmarks SPY,QQQ,VTI
python stock_analysis.py BTC-USD --years 1 --benchmark-preset crypto
python stock_analysis.py AAPL --years 1 --benchmark-preset off
```

Choose the benchmark panel line mode or rolling-correlation window:

```powershell
python stock_analysis.py AAPL --years 1 --benchmark-display cumulative
python stock_analysis.py AAPL --years 1 --benchmark-display all
python stock_analysis.py AAPL --years 1 --benchmark-display excess --corr-window 30
```

Hide excess-return lines when the benchmark panel gets crowded:

```powershell
python stock_analysis.py AAPL --years 1 --no-excess-line
```

Show benchmark lines in the legend:

```powershell
python stock_analysis.py AAPL --years 1 --chart-mode full --show-benchmark-legend
```

### Fixed Historical Periods

Analyze a specific historical period:

```powershell
python stock_analysis.py AAPL --start 2024-01-01 --end 2024-03-15
```

Fixed historical periods are useful for stable tests and repeatable comparisons. Latest analysis should use `--years`.

### Debugging Data Sources

Use these when benchmark rows look suspiciously identical:

```powershell
python stock_analysis.py AAPL --years 1 --no-chart --debug-benchmarks
python stock_analysis.py AAPL --years 1 --no-chart --debug-benchmarks --debug-data-source
python stock_analysis.py AAPL --years 1 --no-chart --debug-benchmarks --debug-data-source --save-debug-columns --output-dir debug
```

`--debug-benchmarks` prints each benchmark's first close, last close, and row count.
`--debug-data-source` also prints yfinance's raw column sample and normalized column sample for global benchmarks.
`--save-debug-columns` writes the full raw yfinance column labels to `<ticker>_yfinance_columns.txt` in `--output-dir`.

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

`OK (skipped=2)` is still a passing result in the default local setup.
The skipped tests are optional checks that need live network access or a local Plotly installation.

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
python stock_analysis.py AAPL --years 1 --show-marker-legend
python stock_analysis.py AAPL --years 1 --hide-signal-markers
```

Open the generated HTML and capture a screenshot. Check that the top summary separates current state, performance, benchmark, strategy, and any data-warning lines; the candlestick chart is not blank; benchmark lines appear in the taller bottom panel; and `full` mode shows grouped legend entries for price, indicators, and benchmarks.
For Korean stocks with data warnings, confirm that the top summary shows the warning line in a distinct accent color.
Compare clean and full screenshots before changing legend position or top margin, because those settings are intentionally conservative defaults.

## GitHub Upload

If an automated environment cannot write to `.git` or cannot reach GitHub, run the final Git commands directly from your local PowerShell:

1. Run tests before saving the change:

```powershell
python -m unittest -v
```

2. Check which files changed:

```powershell
git status
```

3. Stage the files you want to include in the commit:

```powershell
git add .gitignore README.md requirements.txt stock_analysis.py test_stock_analysis.py
```

4. Save the staged changes as a commit:

```powershell
git commit -m "Add backtest tests and README examples"
```

5. Push the commit to GitHub:

```powershell
git push -u origin main
```

## Notes

- Market data depends on external data providers and may change over time.
- Generated dashboard HTML files are intentionally ignored by Git.
- Benchmark comparison uses `SPY` and `QQQ` as broad U.S. market references. For Korean stocks or crypto assets, interpret this as a rough external comparison rather than a perfect peer benchmark.
