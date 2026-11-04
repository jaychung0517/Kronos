
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor


MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

LOOKBACK = 400
HORIZON = 5
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

TZ = ZoneInfo("Asia/Taipei")


def fetch_finmind():
    """下載 FinMind TX 各月份、各時段日行情。"""
    end = datetime.now(TZ).date()
    start = end - timedelta(days=900)

    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": str(start),
        "end_date": str(end),
    }

    token = os.getenv("FINMIND_TOKEN", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = "https://api.finmindtrade.com/api/v4/data"

    for attempt in range(3):
        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=90,
            )
            r.raise_for_status()
            payload = r.json()

            if payload.get("status") != 200:
                raise RuntimeError(
                    f"FinMind API error: {payload}"
                )

            rows = payload.get("data", [])
            if not rows:
                raise RuntimeError(
                    "FinMind returned empty TX data"
                )

            return pd.DataFrame(rows)

        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    raise RuntimeError("FinMind request failed")


def prepare_tx(raw):
    """合併日夜盤，建立近月連續合約。"""
    required = {
        "date", "futures_id", "contract_date",
        "trading_session", "open", "max",
        "min", "close", "volume",
    }

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"TX missing columns: {missing}")

    df = raw.copy()
    df = df[df["futures_id"] == "TX"].copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "max", "min", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 排除價差組合契約，只保留 YYYYMM
    df["contract_date"] = df["contract_date"].astype(str)
    df = df[
        df["contract_date"].str.fullmatch(r"\d{6}")
    ].copy()

    df = df[
        df["trading_session"].isin(
            ["position", "after_market"]
        )
    ]

    df = df[
        (df[["open", "max", "min", "close"]] > 0).all(axis=1)
        & (df["volume"] > 0)
        & (df["max"] >= df[["open", "close", "min"]].max(axis=1))
        & (df["min"] <= df[["open", "close", "max"]].min(axis=1))
    ].copy()

    if df.empty:
        raise RuntimeError("No valid TX records")

    # 同日同月份各時段資料應唯一
    keys = ["date", "contract_date", "trading_session"]
    if df.duplicated(keys).any():
        raise RuntimeError("Duplicate TX session records")

    # 只保留同時具有日盤、夜盤的完整交易日
    counts = df.groupby(
        ["date", "contract_date"]
    )["trading_session"].nunique()

    complete = counts[counts == 2].index

    df = df.set_index(
        ["date", "contract_date"]
    ).loc[complete].reset_index()

    # 合併同一交易日、同一月份
    rows = []

    for (date, contract), group in df.groupby(
        ["date", "contract_date"]
    ):
        night = group[
            group["trading_session"] == "after_market"
        ].iloc[0]

        day = group[
            group["trading_session"] == "position"
        ].iloc[0]

        rows.append({
            "date": date,
            "contract_date": contract,
            "open": night["open"],
            "high": max(night["max"], day["max"]),
            "low": min(night["min"], day["min"]),
            "close": day["close"],
            "volume": night["volume"] + day["volume"],
        })

    combined = pd.DataFrame(rows)

    if combined.empty:
        raise RuntimeError("No complete TX day/night data")

    # 每日選取尚未過期的最近到期月份
    combined["date_month"] = (
        combined["date"].dt.strftime("%Y%m")
    )

    combined = combined[
        combined["contract_date"] >= combined["date_month"]
    ].copy()

    combined = combined.sort_values(
        ["date", "contract_date"]
    )

    selected = (
        combined.groupby("date", as_index=False)
        .first()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(selected) < LOOKBACK:
        raise RuntimeError(
            f"TX only {len(selected)} days, "
            f"need {LOOKBACK}"
        )

    # 轉倉檢查：不能將未調整的合約跳空
    # 當成一般價格變化。
    roll = selected["contract_date"].ne(
        selected["contract_date"].shift()
    )

    if roll.iloc[1:].any():
        raise RuntimeError(
            "TX contract rolls detected. "
            "Back-adjustment must be completed "
            "before production forecasting."
        )

    return selected.set_index("date")[
        ["open", "high", "low", "close"]
    ].tail(LOOKBACK)


def fetch_taiex():
    """下載加權指數歷史資料。"""
    df = yf.download(
        "^TWII",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        raise RuntimeError("TAIEX download failed")

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    cols = ["open", "high", "low", "close"]
    df = df[cols].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if len(df) < LOOKBACK:
        raise RuntimeError("Insufficient TAIEX history")

    return df.tail(LOOKBACK)


def validate_market(df, name):
    cols = ["open", "high", "low", "close"]

    if df.index.has_duplicates:
        raise RuntimeError(f"{name}: duplicate dates")

    if not df.index.is_monotonic_increasing:
        raise RuntimeError(f"{name}: unsorted dates")

    if not np.isfinite(df[cols].to_numpy()).all():
        raise RuntimeError(f"{name}: invalid prices")

    if (df[cols] <= 0).any().any():
        raise RuntimeError(f"{name}: nonpositive prices")

    if (
        (df["high"] < df[cols].max(axis=1))
        | (df["low"] > df[cols].min(axis=1))
    ).any():
        raise RuntimeError(f"{name}: invalid OHLC")

    age = (
        datetime.now(TZ).date()
        - df.index[-1].date()
    ).days

    if age > 5 or age < 0:
        raise RuntimeError(
            f"{name}: stale or future data, age={age}"
        )


def forecast(predictor, df, name):
    """
    測試版使用平日預測日期。
    正式版必須接入交易所交易日曆。
    """
    dates = pd.bdate_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=HORIZON,
    )

    with torch.inference_mode():
        pred = predictor.predict(
            df=df.reset_index(drop=True),
            x_timestamp=pd.Series(df.index),
            y_timestamp=pd.Series(dates),
            pred_len=HORIZON,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

    pred.index = dates
    pred.index.name = "date"

    last_close = float(df["close"].iloc[-1])

    pred["return_pct"] = (
        pred["close"] / last_close - 1
    ) * 100

    pred.to_csv(
        OUTPUT / f"{name}_forecast.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    history = df["close"].tail(60)
    ax.plot(history.index, history.values, label="History")

    line = pd.concat([
        pd.Series(
            [last_close],
            index=[df.index[-1]],
        ),
        pred["close"],
    ])

    ax.plot(
        line.index,
        line.values,
        "--o",
        label="Kronos-base forecast",
    )

    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"{name} forecast")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{name}_forecast.png",
        dpi=150,
    )
    plt.close(fig)

    return {
        "market": name,
        "last_date": str(df.index[-1].date()),
        "last_close": last_close,
        "day_1_pct": float(pred["return_pct"].iloc[0]),
        "day_3_pct": float(pred["return_pct"].iloc[2]),
        "day_5_pct": float(pred["return_pct"].iloc[4]),
    }


def main():
    started = time.time()

    try:
        print("Downloading TAIEX...")
        taiex = fetch_taiex()

        print("Downloading TX...")
        raw_tx = fetch_finmind()
        tx = prepare_tx(raw_tx)

        validate_market(taiex, "TAIEX")
        validate_market(tx, "TX")

        print("Loading Kronos-base...")
        tokenizer = KronosTokenizer.from_pretrained(
            TOKENIZER_NAME
        )
        model = Kronos.from_pretrained(MODEL_NAME)

        predictor = KronosPredictor(
            model,
            tokenizer,
            device="cpu",
            max_context=512,
        )

        results = []

        for name, data in [
            ("TAIEX", taiex),
            ("TX", tx),
        ]:
            print(f"Predicting {name}...")
            results.append(
                forecast(predictor, data, name)
            )

        status = {
            "success": True,
            "model": MODEL_NAME,
            "results": results,
            "elapsed_seconds": round(
                time.time() - started, 2
            ),
        }

        (OUTPUT / "status.json").write_text(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(json.dumps(
            status, ensure_ascii=False, indent=2
        ))

    except Exception as exc:
        error = {
            "success": False,
            "model": MODEL_NAME,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        (OUTPUT / "status.json").write_text(
            json.dumps(
                error, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

        raise


if __name__ == "__main__":
    main()
