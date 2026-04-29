import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import stock_analysis as sa


class StockAnalysisPureLogicTests(unittest.TestCase):
    @staticmethod
    def _sample_price_frame(rows: int = 130) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=rows, freq="B")
        close = np.linspace(100.0, 130.0, rows)
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.arange(rows) + 1000,
                "Adj Close": close,
            },
            index=dates,
        )

    def test_normalize_columns_handles_common_variants(self):
        df = pd.DataFrame(
            {
                " open ": [10.0, 11.0],
                "HIGH": [12.0, 13.0],
                "Low": [9.0, 10.0],
                "Adj Close": [10.5, 12.5],
                "Close": [11.0, 12.0],
                "Vol": [1000, 1100],
            }
        )

        result = sa._normalize_columns(df)

        self.assertEqual(
            list(result.columns),
            ["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        )
        self.assertEqual(result["Open"].tolist(), [10.0, 11.0])
        self.assertEqual(result["Volume"].tolist(), [1000, 1100])

    def test_compute_returns_reports_drawdown_and_annualization_days(self):
        df = pd.DataFrame({"Close": [100.0, 110.0, 99.0, 120.0]})

        result = sa.compute_returns(df, risk_free_rate=0.0, annualization_days=252)

        self.assertIn("daily_return", result)
        self.assertIn("cumulative_return", result)
        self.assertEqual(result["annualization_days"], 252)
        self.assertLess(result["mdd"], 0)
        self.assertGreater(result["annual_return"], 0)

    def test_compute_rsi_keeps_warmup_nan_and_flat_price_neutral(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({"Close": [100.0] * 20}, index=dates)

        result = sa.compute_rsi(df.copy(), period=14)

        self.assertTrue(result["RSI"].iloc[:14].isna().all())
        self.assertEqual(result["RSI"].iloc[-1], 50.0)
        self.assertEqual(result["RSI_Signal"].iloc[-1], "neutral")

    def test_add_indicators_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame(
            {"Close": np.linspace(100.0, 130.0, 80)},
            index=pd.date_range("2024-01-01", periods=80, freq="B"),
        )

        result = sa.add_indicators(df)

        self.assertNotIn("MA20", df.columns)
        self.assertIn("MA20", result.columns)
        self.assertIn("RSI", result.columns)
        self.assertIn("MACD", result.columns)

    def test_main_passes_no_chart_cli_option(self):
        argv = ["stock_analysis.py", "BTC-USD", "--years", "1", "--no-chart"]

        with patch("sys.argv", argv), patch.object(sa, "run_analysis") as run_mock:
            sa.main()

        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["ticker"], "BTC-USD")
        self.assertEqual(kwargs["period_years"], 1.0)
        self.assertFalse(kwargs["show_chart"])

    @unittest.skipUnless(sa._plotly_ok, "plotly is not installed")
    def test_plot_dashboard_returns_plotly_figure(self):
        df = sa.add_indicators(self._sample_price_frame())
        stats = sa.compute_returns(df, risk_free_rate=0.0, annualization_days=252)

        fig = sa.plot_dashboard(df, "TEST", stats)

        self.assertEqual(fig.layout.height, 1000)
        self.assertGreaterEqual(len(fig.data), 8)

    @unittest.skipUnless(
        os.getenv("RUN_NETWORK_TESTS") == "1",
        "set RUN_NETWORK_TESTS=1 to run external data tests",
    )
    @unittest.skipUnless(sa._yfinance_ok, "yfinance is not installed")
    def test_fetch_data_global_smoke(self):
        df = sa.fetch_data("AAPL", start="2024-01-01", end="2024-02-01")

        self.assertFalse(df.empty)
        self.assertEqual(
            list(df.columns),
            ["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        )


if __name__ == "__main__":
    unittest.main()
