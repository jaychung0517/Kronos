"""Independent offline verification for V3.5-A historical data audit."""
import unittest
from datetime import date
import pandas as pd
from scripts.kronos_v35_data_audit import (
    normalize_yahoo, valid_ohlc_rows, validate_tx, tx_front_sessions,
    cross_market_asof,
)


class AuditTests(unittest.TestCase):
    def test_yahoo_ohlc_normalization(self):
        data = pd.DataFrame({
            "Open": [100, 104], "High": [103, 105],
            "Low": [99, 102], "Close": [101, 103],
        }, index=pd.DatetimeIndex(["2021-01-04", "2021-01-05"]))
        out = normalize_yahoo(data, "NASDAQ")
        self.assertEqual(out.loc["2021-01-04", "close"], 101)
        self.assertEqual(len(out), 2)

    def test_invalid_ohlc_rejected(self):
        invalid = pd.DataFrame({
            "open": [100], "high": [99], "low": [98], "close": [101],
        })
        self.assertFalse(valid_ohlc_rows(invalid).iloc[0])

    def test_tx_front_complete_and_missing_night(self):
        rows = []
        for trading_day, session in (
            ("2017-05-16", "position"),
            ("2017-05-16", "after_market"),
            ("2017-05-17", "position"),
        ):
            rows.append({
                "date": trading_day, "futures_id": "TX", "contract_date": "201705",
                "trading_session": session, "open": 100, "max": 102,
                "min": 99, "close": 101, "volume": 20,
            })
        rows.append({**rows[0], "date": "2017-05-18", "volume": -1})
        valid, stats = validate_tx(pd.DataFrame(rows))
        self.assertEqual(stats["invalid_rows"], 1)
        front = tx_front_sessions(valid, date(2017, 5, 16), date(2017, 5, 18))
        self.assertEqual(len(front), 2)
        self.assertEqual(int(front.complete.sum()), 1)

    def test_duplicate_quote_excluded_and_reported(self):
        row = {"date": "2026-11-06", "futures_id": "TX",
               "contract_date": "202611", "trading_session": "position",
               "open": 100, "max": 102, "min": 99, "close": 101, "volume": 20}
        valid, info = validate_tx(pd.DataFrame([row, dict(row)]))
        self.assertEqual(info["duplicate_rows"], 2)
        self.assertEqual(len(valid), 0)

    def test_asof_alignment_no_future_us_close(self):
        taiwan = pd.DatetimeIndex(["2026-11-06"])
        usa = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
        result = cross_market_asof(taiwan, usa, usa)
        for decision in ("TW_CLOSE", "US_CLOSE"):
            row = result[result.decision == decision].iloc[0]
            self.assertEqual(row.NASDAQ_last_available, "2026-11-05")
            self.assertEqual(row.SOX_last_available, "2026-11-05")

    def test_dst_alignment(self):
        # Mar 9 2026 NY market closes 04:00 next-day Taipei due to DST.
        result = cross_market_asof(
            pd.DatetimeIndex(["2026-03-10"]),
            pd.DatetimeIndex(["2026-03-09"]),
            pd.DatetimeIndex(["2026-03-09"]),
        )
        row = result[result.decision == "US_CLOSE"].iloc[0]
        self.assertEqual(row.SOX_last_available, "2026-03-09")


if __name__ == "__main__":
    unittest.main()
