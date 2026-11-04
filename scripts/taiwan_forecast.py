"""Kronos-base: TAIEX and TX combined day + night daily forecast.

Run from the official Kronos repository root:
    python -m scripts.taiwan_forecast

Dependencies beyond upstream Kronos requirements: yfinance, requests, matplotlib.
FinMind dataset: TaiwanFuturesDaily, data_id=TX.

This program is for research; continuous futures use backward additive adjustments.
Future dates fall back to weekdays if data/trading_calendar.csv is not provided.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf

from model import Kronos, KronosTokenizer, KronosPredictor


TZ = ZoneInfo("Asia/Taipei")
MODEL_NAME = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-base")
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK = int(os.getenv("LOOKBACK", "400"))
HORIZON = int(os.getenv("HORIZON", "5"))
SAMPLE_COUNT = int(os.getenv("SAMPLE_COUNT", "1"))
MAX_STALE_DAYS = int(os.getenv("MAX_STALE_DAYS", "7"))
MIN_NIGHT_COVERAGE = float(os.getenv("MIN_NIGHT_COVERAGE", "0.97"))
MAX_OVERLAP_AGE_DAYS = int(os.getenv("MAX_OVERLAP_AGE_DAYS", "10"))
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
OUTPUT = Path("outputs")
CALENDAR_FILE = Path("data/trading_calendar.csv")
OHLC = ["open", "high", "low", "close"]


def log(message: str) -> None:
    print(message, flush=True)


def write_json(name: str, value: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def validate_ohlc(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label}: no records")
    if any(c not in df.columns for c in OHLC):
        raise ValueError(f"{label}: missing OHLC columns")
    nums = df[OHLC].to_numpy(dtype="float64")
    if not np.isfinite(nums).all() or (nums <= 0).any():
        raise ValueError(f"{label}: OHLC has invalid/nonpositive values")
    if (df["high"] < df[OHLC].max(axis=1)).any():
        raise ValueError(f"{label}: high below another OHLC value")
    if (df["low"] > df[OHLC].min(axis=1)).any():
        raise ValueError(f"{label}: low above another OHLC value")


def request_finmind(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": start_date,
        "end_date": end_date,
    }
    token = os.getenv("FINMIND_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                FINMIND_URL, params=params, headers=headers, timeout=90
            )
            # HTTP 402/403: user/action required, not a temporary server failure
            if response.status_code in (401, 402, 403):
                raise RuntimeError(
                    f"FinMind authorization/quota rejected: HTTP {response.status_code}; "
                    "check FINMIND_TOKEN and account plan"
                )
