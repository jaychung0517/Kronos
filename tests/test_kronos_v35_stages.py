"""V3.5-B isolated offline tests; no market network or model inference."""
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.kronos_v35_backtest import (
    session_times, last_market_date, close_time,
    make_contract_bars, eligible_origins,
)
from scripts.kronos_v35_gate import verify, summarize


class V35StageTests(unittest.TestCase):
    def test_us_close_weekend_friday(self):
        usa = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
        events = session_times(pd.Timestamp("2026-11-06"),
                               pd.Timestamp("2026-11-09"), usa)
        self.assertEqual([name for name, _ in events], ["TW_CLOSE", "US_CLOSE"])
        self.assertEqual(events[-1][1].strftime("%Y-%m-%d %H:%M"),
                         "2026-11-07 06:20")

    def test_tx_holiday_advance_suppresses_stale_tw_us_close(self):
        usa = pd.DatetimeIndex(["2021-04-05"])
        tx = pd.DatetimeIndex(["2021-04-01", "2021-04-06"])
        events = session_times(pd.Timestamp("2021-04-01"),
                               pd.Timestamp("2021-04-07"), usa, tx)
        self.assertEqual([name for name, _ in events], ["TW_CLOSE"])

    def test_tx_stays_at_tw_origin_allows_us_close(self):
        usa = pd.DatetimeIndex(["2026-11-06"])
        tx = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
        events = session_times(pd.Timestamp("2026-11-06"),
                               pd.Timestamp("2026-11-09"), usa, tx)
        self.assertEqual([name for name, _ in events], ["TW_CLOSE", "US_CLOSE"])

    def test_us_close_not_created_without_new_us_session(self):
        usa = pd.DatetimeIndex(["2026-11-05"])
        events = session_times(pd.Timestamp("2026-11-06"),
                               pd.Timestamp("2026-11-09"), usa)
        self.assertEqual([s[0] for s in events], ["TW_CLOSE"])

    def test_us_asof_cannot_use_friday_close_before_friday_tw_close(self):
        usa = pd.DatetimeIndex(["2026-11-05", "2026-11-06"])
        friday_tw_close = datetime(2026, 11, 6, 17, 10, tzinfo=ZoneInfo("Asia/Taipei"))
        sat_us_report = datetime(2026, 11, 7, 6, 20, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(last_market_date("NASDAQ", usa, friday_tw_close),
                         pd.Timestamp("2026-11-05"))
        self.assertEqual(last_market_date("NASDAQ", usa, sat_us_report),
                         pd.Timestamp("2026-11-06"))

    def test_dst_session_after_us_close(self):
        usa = pd.DatetimeIndex(["2026-03-09"])
        close = close_time("NASDAQ", usa[0])
        self.assertEqual(close.strftime("%Y-%m-%d %H:%M"), "2026-03-10 04:00")

    def test_tx_contract_day_and_night_merge(self):
        rows = []
        for session, op, hi, lo, cl in [
            ("after_market", 101, 105, 95, 99),
            ("position", 98, 108, 97, 103),
        ]:
            rows.append({"date": "2026-11-06", "contract_date": "202611",
                         "trading_session": session,
                         "open": op, "max": hi, "min": lo, "close": cl,
                         "volume": 100})
        frame = make_contract_bars(pd.DataFrame(rows))
        self.assertEqual(len(frame), 1)
        self.assertEqual(float(frame.iloc[0].open), 101)
        self.assertEqual(float(frame.iloc[0].high), 108)
        self.assertEqual(float(frame.iloc[0].low), 95)
        self.assertEqual(float(frame.iloc[0].close), 103)
        self.assertTrue(bool(frame.iloc[0].has_night))

    def test_gated_rank_count_avoids_lookahead(self):
        origins = pd.bdate_range("2018-01-01", periods=2220)
        front = pd.DataFrame(index=origins, data={"close": np.arange(len(origins))})
        data = eligible_origins({"TAIEX": pd.DataFrame(index=origins)}, front)
        self.assertGreaterEqual(len(data), 2000)
        self.assertTrue(all(data[i] > data[i+1] for i in range(1999)))

    @staticmethod
    def synthetic(n=4):
        rows = []
        start = pd.Timestamp("2026-11-06")
        for rank in range(n):
            origin = (start - pd.Timedelta(days=rank)).date().isoformat()
            for session in ("TW_CLOSE", "US_CLOSE"):
                cutoff = (datetime(2026, 11, 6, 17, 10, tzinfo=ZoneInfo("Asia/Taipei"))
                          - timedelta(days=rank)).isoformat()
                for market in ("TAIEX", "TX", "NASDAQ", "SOX"):
                    for h in (1, 3, 5):
                        for rep in range(3):
                            rows.append({"rank": rank, "tw_origin": origin,
                                "session": session, "market": market,
                                "model_origin": origin, "horizon": h,
                                "seed_index": rep, "random_seed": 1000 + rep,
                                "decision_time_tpe": cutoff,
                                "input_cutoff_tpe": (datetime.fromisoformat(cutoff) -
                                                     timedelta(hours=2)).isoformat(),
                                "pred_return_pct": 1.0,
                                "realized_market_return_pct": 2.0,
                                "realized_taiex_return_pct": 2.0,
                                "path_ohlc_valid": True})
        return pd.DataFrame(rows)

    def test_stage_passes_full_four_market_rows(self):
        rows = self.synthetic()
        status = verify(rows, total=4)
        self.assertEqual(status["tw_close_origins"], 4)
        _, stats, consensus, cs = summarize(rows)
        self.assertGreater(len(stats), 0)
        self.assertTrue(consensus.unanimous.all())
        self.assertTrue(cs.unanimous_correct_pct.eq(1).all())

    def test_stage_rejects_lookahead(self):
        df = self.synthetic()
        df.loc[0, "input_cutoff_tpe"] = (
            datetime.fromisoformat(df.loc[0, "decision_time_tpe"]) +
            timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "LOOKAHEAD"):
            verify(df, total=4)

    def test_stage_rejects_missing_seed(self):
        df = self.synthetic()
        with self.assertRaisesRegex(ValueError, "Missing records"):
            verify(df.iloc[1:], total=4)


if __name__ == "__main__":
    unittest.main()
