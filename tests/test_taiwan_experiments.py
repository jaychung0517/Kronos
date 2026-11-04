"""Unit tests for v2 scenario comparison (no network, no model weights)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import taiwan_forecast as target


class StubPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_p,
                sample_count, verbose):
        self.calls.append((len(df), sample_count, pred_len))
        assert len(df) == len(x_timestamp)
        assert len(y_timestamp) == pred_len
        close = float(df["close"].iloc[-1])
        size = len(df)
        changes = np.array([0.001 * (i + 1) * (sample_count + size / 200)
                            for i in range(pred_len)])
        prices = close * (1 - changes)
        return pd.DataFrame({
            "open": prices * 1.001,
            "high": prices * 1.004,
            "low": prices * 0.996,
            "close": prices,
        })


def make_prices(dates, base):
    values = base + np.arange(len(dates)) * 5.0 + np.sin(np.arange(len(dates))) * 12
    return pd.DataFrame({
        "open": values,
        "high": values + 22,
        "low": values - 21,
        "close": values + 3,
    }, index=dates)


class TaiwanExperimentTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2025-01-02", periods=400)
        self.taiex = make_prices(dates, 21000)
        self.tx = make_prices(dates, 21080)
        self.future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=5)

    def test_four_scenarios_and_report_files(self):
        with tempfile.TemporaryDirectory() as d:
            predictor = StubPredictor()
            with patch.object(target, "OUTPUT", Path(d)):
                results, diagnostics = target.run_experiments(
                    predictor, self.taiex, self.tx, self.future, "weekdays_only"
                )
                self.assertEqual(len(results), 8)
                self.assertEqual(diagnostics["comparisons"], 12)
                self.assertEqual(
                    predictor.calls,
                    [(200, 1, 5)]*2 + [(200, 5, 5)]*2
                    + [(400, 1, 5)]*2 + [(400, 5, 5)]*2,
                )
                summary = pd.read_csv(Path(d) / "experiment_summary.csv")
                self.assertEqual(len(summary), 12)
                self.assertEqual(set(summary["scenario"]), set("ABCD"))
                self.assertEqual(set(summary["day"]), {1, 3, 5})
                self.assertEqual(len(summary[summary["review_status"] == "REVIEW"]),
                                 diagnostics["review_rows"])
                for market in ("TAIEX", "TX"):
                    for s in "ABCD":
                        self.assertTrue((Path(d) / f"{market}_{s}_forecast.csv").exists())
                    self.assertTrue((Path(d) / f"{market}_scenario_comparison.png").exists())
                    self.assertTrue((Path(d) / f"{market}_forecast.csv").exists())
                self.assertIn("參數敏感度", (Path(d) / "experiment_report.md").read_text("utf-8"))

    def test_back_adjust_one_roll(self):
        dates = pd.to_datetime(["2026-01-19", "2026-01-20", "2026-01-21"])
        selected = pd.DataFrame({
            "date": dates,
            "contract_date": ["202601", "202601", "202602"],
            "open": [1000.0, 1005.0, 1050.0],
            "high": [1015.0, 1020.0, 1060.0],
            "low": [990.0, 995.0, 1040.0],
            "close": [1010.0, 1012.0, 1053.0],
        })
        combined = pd.DataFrame({
            "date": [dates[0], dates[1], dates[1], dates[2]],
            "contract_date": ["202601", "202601", "202602", "202602"],
            "close": [1010.0, 1012.0, 1040.0, 1053.0],
        })
        adjusted, roll_log = target.back_adjust_tx(selected, combined)
        self.assertEqual(len(roll_log), 1)
        self.assertEqual(float(roll_log.loc[0, "adjustment_points"]), 28)
        self.assertAlmostEqual(float(adjusted.loc[1, "close"]), 1040)
        self.assertAlmostEqual(float(adjusted.loc[2, "close"]), 1053)

    def test_invalid_forecast_rejected(self):
        bad = pd.DataFrame({
            "open": [100.0], "high": [80.0], "low": [70.0], "close": [90.0]
        })
        with self.assertRaises(ValueError):
            target.validate_ohlc(bad, "bad")


if __name__ == "__main__":
    unittest.main()
