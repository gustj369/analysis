# Project Agent Guide

## 1. Role

You are a careful coding agent working on this repository.
This project focuses on Python-based financial data analysis, technical indicators, backtesting, and Plotly dashboard generation.

## 2. Must Check Before Work

- Confirm whether the user wants analysis only, code changes, or regenerated dashboard files.
- Read the related code before editing, especially `stock_analysis.py`.
- Do not modify generated dashboard HTML files unless explicitly requested.
- Treat market data as time-sensitive when current prices or recent results matter.
- If package installation or network access is needed, explain why before proceeding.

## 3. Core Principles

- Do not make broad changes without explaining the reason.
- Prefer small, reviewable diffs.
- Preserve the existing single-script architecture unless there is a clear reason.
- Before editing, summarize the intended change.
- After editing, explain what changed and how to verify it.
- Keep outputs useful for decision-making, not just technically correct.

## 4. Project Context

- Project purpose: fetch financial market data, calculate indicators and risk/return metrics, run simple strategy backtests, and generate interactive dashboards.
- Main tech stack: Python, pandas, numpy, plotly, yfinance, FinanceDataReader.
- Core files and folders:
  - `stock_analysis.py`: main analysis pipeline and executable entry point.
  - `*_dashboard.html`: generated Plotly dashboard outputs.
  - `.claude/settings.local.json`: legacy Claude Code permission settings.
  - `__pycache__/`: generated Python cache.
- Run method: execute `stock_analysis.py` with a Python interpreter after required packages are installed.
- Test method: no formal test suite is currently present. Prefer small local checks for the changed function, and avoid network-dependent verification unless the task needs live data.

## 5. Coding Style

- Naming rules: keep existing snake_case function and variable names.
- File structure rules: keep changes focused in the relevant file; avoid splitting the project unless requested.
- Error handling: raise clear `ValueError` or `ImportError` for invalid inputs, missing data, or missing dependencies.
- Comments: add comments only when they clarify non-obvious financial logic, data assumptions, or edge-case handling.
- Dashboard code: keep Plotly chart configuration explicit and readable.
- Commit/PR rules: summarize the user-visible behavior change, verification performed, and any data/source assumptions.

## 6. Workflow

1. Read related files first.
2. Summarize understanding.
3. Propose a minimal plan for non-trivial changes.
4. Make small changes.
5. Run or suggest the smallest useful verification.
6. Report changed files and remaining risks.

## 7. Code Editing Checklist

Before editing:

- Identify the exact file, function, or output being changed.
- Check whether the change affects data fetching, calculations, chart rendering, backtesting, or saved outputs.
- Note assumptions about tickers, date ranges, currencies, fees, risk-free rates, or data sources.
- Avoid touching unrelated files.

After editing:

- Verify the edited path with the smallest practical check.
- Confirm generated charts or dashboards still load if output code changed.
- Check that missing or malformed market data produces understandable errors.
- Summarize changed files, verification, and any residual risk.

## 8. Do Not

- Do not make large speculative refactors.
- Do not change existing API contracts without a clear reason.
- Do not change core calculation logic without verification.
- Do not add unnecessary dependencies.
- Do not overwrite existing generated dashboard files unless regeneration is requested.
