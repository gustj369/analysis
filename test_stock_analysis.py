import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import stock_analysis as sa


RUN_ANALYSIS_RESULT_KEYS = {
    "df",
    "stats",
    "bt_result",
    "bah_result",
    "annualization_days",
    "current_state",
    "benchmark_comparison",
    "strategy_summary",
    "data_quality",
    "external_price_check",
}


class StockAnalysisPureLogicTests(unittest.TestCase):
    @staticmethod
    def _sample_price_frame(rows: int = 130,
                            close: np.ndarray | None = None) -> pd.DataFrame:
        """Create deterministic OHLCV test data with an Adj Close column."""
        dates = pd.date_range("2024-01-01", periods=rows, freq="B")
        if close is None:
            close = np.linspace(100.0, 130.0, rows)
        else:
            close = np.asarray(close, dtype=float)
            rows = len(close)
            dates = pd.date_range("2024-01-01", periods=rows, freq="B")
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

    def test_summarize_current_state_reports_latest_signal_score(self):
        df = sa.add_indicators(self._sample_price_frame())

        summary = sa.summarize_current_state(df)

        self.assertEqual(summary["trend"], "상승 우위")
        self.assertGreaterEqual(summary["signal_score"], 4)
        self.assertEqual(summary["signal_score_max"], 5)
        self.assertTrue(summary["price_position"]["above_ma20"])
        self.assertTrue(summary["price_position"]["above_ma60"])
        self.assertIn(summary["rsi_state"], {"중립~강세", "과매수"})
        self.assertIn("모멘텀", summary["macd_state"])

    def test_main_passes_no_chart_cli_option(self):
        argv = ["stock_analysis.py", "BTC-USD", "--years", "1", "--no-chart"]

        with patch("sys.argv", argv), patch.object(sa, "run_analysis") as run_mock:
            sa.main()

        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["ticker"], "BTC-USD")
        self.assertEqual(kwargs["period_years"], 1.0)
        self.assertFalse(kwargs["show_chart"])
        self.assertEqual(kwargs["benchmark_display"], "cumulative")
        self.assertFalse(kwargs["show_marker_legend"])
        self.assertFalse(kwargs["show_signal_markers"])
        self.assertFalse(kwargs["show_benchmark_legend"])

    def test_main_passes_explicit_cli_options(self):
        argv = [
            "stock_analysis.py",
            "005930",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-01",
            "--capital",
            "5000000",
            "--commission",
            "0.001",
            "--rfr",
            "0.02",
            "--overwrite",
            "--benchmarks",
            "VTI,QQQ",
            "--chart-mode",
            "full",
            "--benchmark-preset",
            "crypto",
            "--no-excess-line",
            "--benchmark-display",
            "excess",
            "--corr-window",
            "30",
            "--show-marker-legend",
            "--hide-signal-markers",
            "--show-benchmark-legend",
            "--debug-data-source",
            "--save-debug-columns",
        ]

        with patch("sys.argv", argv), patch.object(sa, "run_analysis") as run_mock:
            sa.main()

        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["ticker"], "005930")
        self.assertEqual(kwargs["start"], "2024-01-01")
        self.assertEqual(kwargs["end"], "2024-03-01")
        self.assertEqual(kwargs["initial_capital"], 5_000_000.0)
        self.assertEqual(kwargs["commission"], 0.001)
        self.assertEqual(kwargs["risk_free_rate"], 0.02)
        self.assertTrue(kwargs["overwrite_html"])
        self.assertEqual(kwargs["benchmarks"], ("VTI", "QQQ"))
        self.assertEqual(kwargs["benchmark_preset"], "crypto")
        self.assertFalse(kwargs["show_excess_return"])
        self.assertEqual(kwargs["benchmark_display"], "excess")
        self.assertEqual(kwargs["corr_window"], 30)
        self.assertTrue(kwargs["show_marker_legend"])
        self.assertFalse(kwargs["show_signal_markers"])
        self.assertTrue(kwargs["show_benchmark_legend"])
        self.assertTrue(kwargs["debug_data_source"])
        self.assertTrue(kwargs["save_debug_columns"])

    def test_full_chart_mode_keeps_benchmark_legend_hidden_by_default(self):
        argv = ["stock_analysis.py", "AAPL", "--chart-mode", "full", "--no-chart"]

        with patch("sys.argv", argv), patch.object(sa, "run_analysis") as run_mock:
            sa.main()

        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["benchmark_display"], "all")
        self.assertFalse(kwargs["show_marker_legend"])
        self.assertTrue(kwargs["show_signal_markers"])
        self.assertFalse(kwargs["show_benchmark_legend"])

    def test_korean_analysis_does_not_require_yfinance_dependency(self):
        with patch.object(sa, "_yfinance_ok", False), patch.object(
            sa, "_fdr_ok", True
        ), patch.object(sa, "_plotly_ok", True):
            sa._check_deps(
                need_plotly=False,
                need_fdr=True,
                need_yfinance=False,
            )

    def test_run_analysis_includes_current_state_summary(self):
        df = self._sample_price_frame()

        with patch.object(sa, "fetch_data", return_value=df), patch.object(
            sa, "_check_deps"
        ), patch.object(sa, "_print_intermediate_analysis_reports"), patch.object(
            sa, "_print_analysis_reports"
        ), patch(
            "sys.stdout", new_callable=StringIO
        ):
            result = sa.run_analysis("TEST", show_chart=False)

        self.assertEqual(set(result), RUN_ANALYSIS_RESULT_KEYS)
        self.assertEqual(result["current_state"]["trend"], "상승 우위")
        self.assertEqual(result["current_state"]["signal_score_max"], 5)
        self.assertEqual([row["ticker"] for row in result["benchmark_comparison"]], ["SPY", "QQQ"])

    def test_run_analysis_delegates_dashboard_save_and_show(self):
        df = self._sample_price_frame()
        fig = Mock()

        with patch.object(sa, "fetch_data", return_value=df), patch.object(
            sa, "_check_deps"
        ), patch.object(sa, "_print_intermediate_analysis_reports"), patch.object(
            sa, "_print_analysis_reports"
        ), patch.object(sa, "plot_dashboard", return_value=fig) as plot_mock, patch.object(
            sa, "_save_and_show_dashboard", return_value=Path("TEST_dashboard.html")
        ) as save_mock, patch(
            "sys.stdout", new_callable=StringIO
        ):
            result = sa.run_analysis(
                "TEST",
                show_chart=True,
                overwrite_html=True,
                output_dir="reports",
            )

        plot_mock.assert_called_once()
        save_mock.assert_called_once_with(fig, "TEST", True, "reports")
        self.assertIn("df", result)

    def test_run_analysis_delegates_data_validation_reports(self):
        df = self._sample_price_frame()
        data_quality = {"checked": False, "reason": "not_korean_ticker"}
        external_price_check = {"checked": False, "reason": "not_korean_ticker"}

        with patch.object(sa, "fetch_data", return_value=df), patch.object(
            sa, "_check_deps"
        ), patch.object(
            sa,
            "_print_data_validation_reports",
            return_value=(data_quality, external_price_check),
        ) as validation_mock, patch.object(
            sa, "_print_intermediate_analysis_reports"
        ), patch.object(
            sa, "_print_analysis_reports"
        ), patch(
            "sys.stdout", new_callable=StringIO
        ):
            result = sa.run_analysis("TEST", show_chart=False)

        validation_mock.assert_called_once_with("TEST", df)
        self.assertIs(result["data_quality"], data_quality)
        self.assertIs(result["external_price_check"], external_price_check)

    def test_build_analysis_result_returns_core_result_without_report_hooks(self):
        df = self._sample_price_frame()

        with patch.object(sa, "fetch_data", return_value=df), redirect_stdout(StringIO()):
            result = sa._build_analysis_result(
                ticker="TEST",
                period_years=1,
                start=None,
                end=None,
                initial_capital=10_000_000,
                commission=0.00015,
                risk_free_rate=0.03,
                benchmarks=(),
                benchmark_preset="off",
                corr_window=60,
                debug_benchmarks=False,
                debug_data_source=False,
                save_debug_columns=False,
                output_dir=".",
            )

        self.assertIn("df", result)
        self.assertIn("stats", result)
        self.assertIn("bt_result", result)
        self.assertIn("bah_result", result)
        self.assertIn("strategy_summary", result)
        self.assertEqual(result["benchmark_comparison"], [])

    def test_make_cli_report_hooks_contains_expected_steps(self):
        hooks = sa._make_cli_report_hooks()

        self.assertEqual(
            set(hooks),
            {
                "data_validation",
                "annualization",
                "indicators",
                "current_state",
                "benchmark",
            },
        )
        self.assertIs(hooks["data_validation"], sa._print_data_validation_reports)
        self.assertIs(hooks["annualization"], sa._print_annualization_basis)
        self.assertIs(hooks["indicators"], sa._print_indicator_calculation_done)

    def test_run_analysis_delegates_final_report_printing(self):
        df = self._sample_price_frame()

        with patch.object(sa, "fetch_data", return_value=df), patch.object(
            sa, "_check_deps"
        ), patch.object(sa, "_print_intermediate_analysis_reports"), patch.object(
            sa, "_print_analysis_reports"
        ) as report_mock, patch(
            "sys.stdout", new_callable=StringIO
        ):
            result = sa.run_analysis("TEST", show_chart=False)

        report_mock.assert_called_once_with(
            result["bt_result"], result["bah_result"], "TEST"
        )

    def test_run_analysis_delegates_intermediate_report_printing(self):
        df = self._sample_price_frame()

        with patch.object(sa, "fetch_data", return_value=df), patch.object(
            sa, "_check_deps"
        ), patch.object(
            sa, "_print_intermediate_analysis_reports"
        ) as report_mock, patch.object(
            sa, "_print_analysis_reports"
        ), patch(
            "sys.stdout", new_callable=StringIO
        ):
            result = sa.run_analysis("TEST", show_chart=False)

        self.assertEqual(report_mock.call_count, 2)
        first_call = report_mock.call_args_list[0]
        second_call = report_mock.call_args_list[1]
        self.assertEqual(first_call.args, ("TEST",))
        self.assertEqual(first_call.kwargs["current_state"], result["current_state"])
        self.assertEqual(second_call.args, ("TEST",))
        self.assertEqual(second_call.kwargs["stats"], result["stats"])
        self.assertEqual(
            second_call.kwargs["benchmark_comparison"],
            result["benchmark_comparison"],
        )

    def test_compute_benchmark_comparison_reports_excess_return(self):
        target_df = self._sample_price_frame(close=np.array([100.0, 120.0]))
        benchmark_df = self._sample_price_frame(close=np.array([100.0, 110.0]))
        target_stats = sa.compute_returns(target_df, risk_free_rate=0.0, annualization_days=252)

        with patch.object(sa, "fetch_data", return_value=benchmark_df), redirect_stdout(StringIO()):
            rows = sa.compute_benchmark_comparison(
                target_stats,
                start="2024-01-01",
                end="2024-01-03",
                risk_free_rate=0.0,
                annualization_days=252,
                benchmarks=("SPY",),
                corr_window=2,
                debug=True,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "SPY")
        self.assertIsNone(rows[0]["error"])
        self.assertAlmostEqual(rows[0]["total_return"], 0.10)
        self.assertAlmostEqual(rows[0]["excess_return"], 0.10)
        self.assertIn("cumulative_return", rows[0])
        self.assertIn("excess_return_series", rows[0])
        self.assertIn("rolling_corr", rows[0])
        self.assertIn("latest_corr", rows[0])
        self.assertEqual(rows[0]["corr_window"], 2)
        self.assertEqual(rows[0]["first_close"], 100.0)
        self.assertEqual(rows[0]["last_close"], 110.0)
        self.assertEqual(rows[0]["n_rows"], 2)

    def test_compute_benchmark_comparison_keeps_distinct_benchmark_results(self):
        target_df = self._sample_price_frame(close=np.array([100.0, 130.0]))
        spy_df = self._sample_price_frame(close=np.array([100.0, 110.0]))
        qqq_df = self._sample_price_frame(close=np.array([100.0, 120.0]))
        target_stats = sa.compute_returns(target_df, risk_free_rate=0.0, annualization_days=252)

        def fake_fetch(ticker, start, end):
            return {"SPY": spy_df, "QQQ": qqq_df}[ticker]

        with patch.object(sa, "fetch_data", side_effect=fake_fetch) as fetch_mock:
            rows = sa.compute_benchmark_comparison(
                target_stats,
                start="2024-01-01",
                end="2024-01-03",
                risk_free_rate=0.0,
                annualization_days=252,
                benchmarks=("SPY", "QQQ"),
                corr_window=2,
            )

        self.assertEqual([call.args[0] for call in fetch_mock.call_args_list], ["SPY", "QQQ"])
        self.assertEqual([row["ticker"] for row in rows], ["SPY", "QQQ"])
        self.assertNotEqual(rows[0]["total_return"], rows[1]["total_return"])
        self.assertAlmostEqual(rows[0]["total_return"], 0.10)
        self.assertAlmostEqual(rows[1]["total_return"], 0.20)

    def test_compute_benchmark_comparison_debug_warns_on_duplicate_price_signature(self):
        target_df = self._sample_price_frame(close=np.array([100.0, 130.0]))
        benchmark_df = self._sample_price_frame(close=np.array([100.0, 110.0]))
        target_stats = sa.compute_returns(target_df, risk_free_rate=0.0, annualization_days=252)

        stdout = StringIO()
        with patch.object(sa, "fetch_data", return_value=benchmark_df), redirect_stdout(stdout):
            rows = sa.compute_benchmark_comparison(
                target_stats,
                start="2024-01-01",
                end="2024-01-03",
                benchmarks=("SPY", "QQQ"),
                debug=True,
            )

        self.assertEqual([row["ticker"] for row in rows], ["SPY", "QQQ"])
        self.assertIn("벤치마크 검증", stdout.getvalue())
        self.assertIn("벤치마크 경고", stdout.getvalue())

    def test_compute_benchmark_debug_source_uses_single_fetch_path(self):
        target_df = self._sample_price_frame(close=np.array([100.0, 130.0]))
        benchmark_df = self._sample_price_frame(close=np.array([100.0, 110.0]))
        target_stats = sa.compute_returns(target_df, risk_free_rate=0.0, annualization_days=252)

        with patch.object(sa, "_fetch_global") as raw_fetch_mock, patch.object(
            sa, "fetch_data", return_value=benchmark_df
        ) as fetch_mock:
            rows = sa.compute_benchmark_comparison(
                target_stats,
                start="2024-01-01",
                end="2024-01-03",
                benchmarks=("SPY",),
                debug_source=True,
                debug_columns_dir="debug",
            )

        raw_fetch_mock.assert_not_called()
        fetch_mock.assert_called_once_with(
            "SPY",
            start="2024-01-01",
            end="2024-01-03",
            debug_source=True,
            debug_columns_dir="debug",
        )
        self.assertEqual(rows[0]["ticker"], "SPY")

    def test_flatten_yfinance_columns_selects_ohlcv_level(self):
        columns = pd.MultiIndex.from_product(
            [["AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        df = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=columns)

        flattened = sa._flatten_yfinance_columns(df, "AAPL")

        self.assertEqual(
            list(flattened.columns),
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )

    def test_fetch_global_debug_source_prints_column_samples(self):
        raw = self._sample_price_frame(close=np.array([100.0, 110.0]))
        fake_yf = Mock()
        fake_yf.download.return_value = raw

        stdout = StringIO()
        with patch.object(sa, "yf", fake_yf), redirect_stdout(stdout):
            result = sa._fetch_global(
                "SPY",
                start="2024-01-01",
                end="2024-01-03",
                debug_source=True,
            )

        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume", "Adj Close"])
        self.assertIn("데이터 원천", stdout.getvalue())
        self.assertIn("normalized columns", stdout.getvalue())

    def test_fetch_global_can_save_full_raw_columns(self):
        fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ticker = "TEST-SAVE"
        columns = pd.MultiIndex.from_tuples([(ticker, field) for field in fields])
        raw = pd.DataFrame(
            [[100.0, 101.0, 99.0, 100.5, 100.5, 1000]],
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
            columns=columns,
        )
        fake_yf = Mock()
        fake_yf.download.return_value = raw
        out_path = Path("TEST-SAVE_yfinance_columns.txt")
        if out_path.exists():
            out_path.unlink()

        try:
            stdout = StringIO()
            with patch.object(sa, "yf", fake_yf), redirect_stdout(stdout):
                result = sa._fetch_global(
                    ticker,
                    start="2024-01-01",
                    end="2024-01-03",
                    debug_source=True,
                    debug_columns_dir=".",
                )

            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("('TEST-SAVE', 'Open')", content)
            self.assertIn("full raw columns saved", stdout.getvalue())
            self.assertEqual(list(result.columns), fields)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_compute_benchmark_comparison_keeps_fetch_errors(self):
        target_df = self._sample_price_frame(close=np.array([100.0, 120.0]))
        target_stats = sa.compute_returns(target_df, risk_free_rate=0.0, annualization_days=252)

        with patch.object(sa, "fetch_data", side_effect=ValueError("data unavailable")):
            rows = sa.compute_benchmark_comparison(
                target_stats,
                start="2024-01-01",
                end="2024-01-03",
                benchmarks=("SPY",),
            )

        self.assertEqual(rows[0]["ticker"], "SPY")
        self.assertIsNone(rows[0]["total_return"])
        self.assertIsNone(rows[0]["cumulative_return"])
        self.assertIn("data unavailable", rows[0]["error"])

    def test_default_benchmarks_follow_asset_type(self):
        self.assertEqual(sa._default_benchmarks_for_ticker("AAPL"), ("SPY", "QQQ"))
        self.assertEqual(sa._default_benchmarks_for_ticker("005930"), ("EWY", "SPY"))
        self.assertEqual(sa._default_benchmarks_for_ticker("BTC-USD"), ("ETH-USD",))
        self.assertEqual(sa._select_benchmarks("ETH-USD", preset="crypto"), ("BTC-USD",))
        self.assertEqual(sa._select_benchmarks("AAPL", preset="off"), ())

    def test_print_data_quality_check_reports_korean_stock_source(self):
        df = self._sample_price_frame(close=np.array([100.0, 110.0, 120.0]))

        stdout = StringIO()
        with redirect_stdout(stdout):
            result = sa.print_data_quality_check("005930", df)

        output = stdout.getvalue()
        self.assertTrue(result["checked"])
        self.assertEqual(result["ticker"], "005930")
        self.assertEqual(result["latest_close"], 120.0)
        self.assertEqual(result["warnings"], [])
        self.assertIn("close_range_ratio", result)
        self.assertIn("max_abs_daily_return", result)
        self.assertIn("데이터 검증", output)
        self.assertIn("005930", output)

    def test_print_data_quality_check_flags_unusual_korean_price_moves(self):
        df = self._sample_price_frame(close=np.array([100.0, 105.0, 600.0]))

        stdout = StringIO()
        with redirect_stdout(stdout):
            result = sa.print_data_quality_check("005930", df)

        self.assertIn("wide_close_range", result["warnings"])
        self.assertIn("latest_close_outlier", result["warnings"])
        self.assertIn("large_daily_move", result["warnings"])
        self.assertIn("경고", stdout.getvalue())

    def test_print_external_price_check_compares_yfinance_reference(self):
        df = self._sample_price_frame(close=np.array([100.0, 110.0, 120.0]))
        ref_df = self._sample_price_frame(close=np.array([100.0, 110.0, 121.0]))

        stdout = StringIO()
        with patch.object(sa, "_yfinance_ok", True), patch.object(
            sa, "_fetch_global", return_value=ref_df
        ), redirect_stdout(stdout):
            result = sa.print_external_price_check("005930", df)

        output = stdout.getvalue()
        self.assertTrue(result["checked"])
        self.assertEqual(result["reference_ticker"], "005930.KS")
        self.assertAlmostEqual(result["diff_pct"], (120.0 / 121.0 - 1) * 100)
        self.assertIn("외부 기준 가격", output)
        self.assertIn("005930.KS", output)

    def test_print_external_price_check_reports_reference_failures(self):
        df = self._sample_price_frame(close=np.array([100.0, 110.0, 120.0]))

        stdout = StringIO()
        with patch.object(sa, "_yfinance_ok", True), patch.object(
            sa, "_fetch_global", side_effect=ValueError("no reference data")
        ), redirect_stdout(stdout):
            result = sa.print_external_price_check("005930", df)

        output = stdout.getvalue()
        self.assertFalse(result["checked"])
        self.assertEqual(result["reason"], "reference_not_found")
        self.assertIn("005930.KS", result["failures"])
        self.assertIn("005930.KQ", result["failures"])
        self.assertIn("사유", output)

    def test_backtest_golden_cross_returns_expected_shape(self):
        close = np.concatenate([
            np.linspace(120.0, 80.0, 80),
            np.linspace(81.0, 150.0, 80),
        ])
        df = sa.add_indicators(self._sample_price_frame(close=close))

        result = sa.backtest_golden_cross(
            df,
            initial_capital=1_000_000,
            commission=0.0,
            risk_free_rate=0.0,
            annualization_days=252,
        )

        expected_keys = {
            "trades",
            "equity_series",
            "final_equity",
            "total_return",
            "annual_return",
            "annual_vol",
            "sharpe",
            "mdd",
            "n_trades",
            "win_rate",
            "initial_capital",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(len(result["equity_series"]), len(df))
        self.assertGreaterEqual(result["n_trades"], 1)
        self.assertGreater(result["final_equity"], 0)

    def test_backtest_golden_cross_calculates_fixed_trade_result(self):
        df = self._sample_price_frame(close=np.array([100.0, 100.0, 120.0]))
        df["MA_Signal"] = ["", "golden", "dead"]

        result = sa.backtest_golden_cross(
            df,
            initial_capital=1_000.0,
            commission=0.0,
            risk_free_rate=0.0,
            annualization_days=252,
        )

        self.assertEqual(result["n_trades"], 1)
        self.assertEqual(len(result["trades"]), 2)
        self.assertEqual(result["final_equity"], 1_200.0)
        self.assertAlmostEqual(result["total_return"], 0.2)
        self.assertEqual(result["win_rate"], 1.0)

    def test_backtest_buy_and_hold_returns_expected_shape(self):
        df = self._sample_price_frame(close=np.array([100.0, 110.0, 120.0]))

        result = sa.backtest_buy_and_hold(
            df,
            initial_capital=1_000_000,
            commission=0.0,
            risk_free_rate=0.0,
            annualization_days=252,
        )

        self.assertEqual(result["initial_capital"], 1_000_000)
        self.assertEqual(result["final_equity"], 1_200_000)
        self.assertAlmostEqual(result["total_return"], 0.2)
        self.assertIn("annual_return", result)
        self.assertIn("mdd", result)

    def test_backtest_buy_and_hold_applies_commission(self):
        df = self._sample_price_frame(close=np.array([100.0, 120.0]))

        result = sa.backtest_buy_and_hold(
            df,
            initial_capital=1_000.0,
            commission=0.01,
            risk_free_rate=0.0,
            annualization_days=252,
        )

        self.assertAlmostEqual(result["final_equity"], 1_176.12)
        self.assertAlmostEqual(result["total_return"], 0.17612)

    def test_main_exits_when_run_analysis_reports_invalid_parameters(self):
        argv = ["stock_analysis.py", "AAPL", "--capital", "-1", "--no-chart"]

        with patch("sys.argv", argv), patch.object(
            sa,
            "run_analysis",
            side_effect=ValueError("initial_capital은 0보다 커야 합니다."),
        ):
            stdout = StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as cm:
                sa.main()

        self.assertEqual(cm.exception.code, 1)
        self.assertIn("[오류]", stdout.getvalue())

    def test_main_exits_on_invalid_cli_type(self):
        argv = ["stock_analysis.py", "AAPL", "--years", "not-a-number"]

        with patch("sys.argv", argv):
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as cm:
                sa.main()

        self.assertNotEqual(cm.exception.code, 0)
        self.assertIn("invalid float value", stderr.getvalue())

    # Optional: requires Plotly, so lightweight environments can still run pure logic tests.
    @unittest.skipUnless(sa._plotly_ok, "plotly is not installed")
    def test_plot_dashboard_returns_plotly_figure(self):
        df = sa.add_indicators(self._sample_price_frame())
        stats = sa.compute_returns(df, risk_free_rate=0.0, annualization_days=252)
        current_state = sa.summarize_current_state(df)

        benchmark_comparison = [{
            "ticker": "SPY",
            "total_return": 0.10,
            "annual_return": 0.20,
            "mdd": -0.05,
            "excess_return": 0.03,
            "cumulative_return": stats["cumulative_return"],
            "excess_return_series": stats["cumulative_return"] - stats["cumulative_return"],
            "rolling_corr": stats["daily_return"].rolling(60).corr(stats["daily_return"]),
            "latest_corr": 1.0,
            "corr_window": 60,
            "error": None,
        }]
        strategy_summary = {
            "winner": "Buy & Hold 전략",
            "return_diff": -0.10,
            "sharpe_diff": -0.25,
            "mdd_diff": 0.02,
        }
        data_quality = {
            "checked": True,
            "warnings": ["wide_close_range"],
        }

        fig = sa.plot_dashboard(
            df,
            "TEST",
            stats,
            current_state=current_state,
            benchmark_comparison=benchmark_comparison,
            strategy_summary=strategy_summary,
            data_quality=data_quality,
        )

        self.assertEqual(fig.layout.height, 1120)
        self.assertEqual(fig.layout.margin.t, 135)  # 5줄 어노테이션 → CHART_MARGIN_LARGE
        self.assertEqual(fig.layout.legend.font.size, 10)
        self.assertGreaterEqual(len(fig.data), 10)
        self.assertTrue(fig.layout.annotations)
        self.assertGreaterEqual(len(fig.layout.annotations), 2)
        self.assertIn("현재 상태", fig.layout.annotations[-1].text)
        self.assertIn("점수", fig.layout.annotations[-1].text)

        annotation_text = " ".join(annotation.text or "" for annotation in fig.layout.annotations)
        self.assertIn("60D", annotation_text)
        self.assertIn("성과", annotation_text)
        self.assertIn("전략", annotation_text)
        self.assertIn("데이터 주의", annotation_text)
        self.assertIn("wide_close_range", annotation_text)
        self.assertIn("#FFB74D", annotation_text)

        benchmark_trace_names = {"TEST 누적수익률", "SPY 누적수익률", "TEST vs SPY 초과수익"}
        benchmark_traces = [trace for trace in fig.data if trace.name in benchmark_trace_names]
        self.assertTrue(benchmark_traces)
        self.assertTrue(all(trace.showlegend is False for trace in benchmark_traces))
        self.assertTrue(all(trace.legendgroup == "benchmark" for trace in benchmark_traces))
        self.assertIn("price", {trace.legendgroup for trace in fig.data if trace.name == "캔들"})
        self.assertIn("indicator", {trace.legendgroup for trace in fig.data if trace.name == "RSI"})

        fig_all_lines = sa.plot_dashboard(
            df,
            "TEST",
            stats,
            current_state=current_state,
            benchmark_comparison=benchmark_comparison,
            benchmark_display="all",
            strategy_summary=strategy_summary,
            show_benchmark_legend=True,
        )
        self.assertEqual(len(fig_all_lines.data), len(fig.data) + 1)
        benchmark_traces = [trace for trace in fig_all_lines.data if trace.name in benchmark_trace_names]
        self.assertTrue(any(trace.showlegend is True for trace in benchmark_traces))

        fig_excess_only = sa.plot_dashboard(
            df,
            "TEST",
            stats,
            current_state=current_state,
            benchmark_comparison=benchmark_comparison,
            benchmark_display="excess",
            strategy_summary=strategy_summary,
        )
        self.assertLess(len(fig_excess_only.data), len(fig.data))

        fig_without_marker_legend = sa.plot_dashboard(
            df,
            "TEST",
            stats,
            current_state=current_state,
            benchmark_comparison=benchmark_comparison,
            show_marker_legend=False,
            show_signal_markers=True,
            strategy_summary=strategy_summary,
        )
        marker_names = {"골든크로스", "데드크로스", "RSI 매수", "RSI 매도", "MACD 골든", "MACD 데드"}
        marker_traces = [trace for trace in fig_without_marker_legend.data if trace.name in marker_names]
        self.assertTrue(marker_traces)
        self.assertTrue(all(trace.showlegend is False for trace in marker_traces))

        fig_without_signal_markers = sa.plot_dashboard(
            df,
            "TEST",
            stats,
            current_state=current_state,
            benchmark_comparison=benchmark_comparison,
            show_signal_markers=False,
            strategy_summary=strategy_summary,
        )
        marker_traces = [trace for trace in fig_without_signal_markers.data if trace.name in marker_names]
        self.assertEqual(marker_traces, [])

    # Optional: requires live network access and yfinance, so it is off by default.
    @unittest.skipUnless(
        os.getenv("RUN_NETWORK_TESTS") == "1",
        "set RUN_NETWORK_TESTS=1 to run external data tests",
    )
    @unittest.skipUnless(sa._yfinance_ok, "yfinance is not installed")
    def test_fetch_data_global_smoke(self):
        # Fixed historical range keeps this optional smoke test stable and
        # long enough to satisfy fetch_data's minimum 30-trading-day guard.
        df = sa.fetch_data("AAPL", start="2024-01-01", end="2024-03-15")

        self.assertFalse(df.empty)
        self.assertEqual(
            list(df.columns),
            ["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        )


if __name__ == "__main__":
    unittest.main()
