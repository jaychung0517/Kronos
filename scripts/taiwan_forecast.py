
"""Kronos-base V3.2: Taiwan market forecasts and multi-seed backtests.

Run from Kronos repository root:
    python -m scripts.taiwan_forecast

Markets:
    TAIEX: Yahoo Finance ^TWII
    TX: FinMind TaiwanFuturesDaily, day + night

Research only. No automated trading.
"""

from __future__ import annotations

import json
import os
import random
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


# ============================================================
# 1. 系統設定
# ============================================================

TZ = ZoneInfo("Asia/Taipei")
OUTPUT = Path("outputs")
CALENDAR = Path("data/trading_calendar.csv")

OHLC = ["open", "high", "low", "close"]

MODEL_NAME = os.getenv(
    "KRONOS_MODEL",
    "NeoQuasar/Kronos-base"
)

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

MODE = os.getenv(
    "KRONOS_MODE", "live"
).strip().lower()

BACKTEST_DAYS = int(
    float(os.getenv("BACKTEST_DAYS", "20"))
)

BASE_SEED = int(
    os.getenv("KRONOS_SEED", "20261104")
)

SEED_COUNT = int(os.getenv(
    "KRONOS_SEED_COUNT",
    "3" if MODE == "backtest" else "1"
))

SCENARIO_CONFIG = {
    "A": (200, 1),
    "B": (200, 5),
    "C": (400, 1),
    "D": (400, 5),
}

# 保留舊版單元測試相容性
SCENARIOS = [
    (key, *value)
    for key, value in SCENARIO_CONFIG.items()
]

SELECTED = os.getenv(
    "KRONOS_SCENARIOS",
    "AB" if MODE == "backtest" else "ABCD"
).strip().upper()

MAX_STALE_DAYS = int(
    os.getenv("MAX_STALE_DAYS", "7")
)

MIN_NIGHT_COVERAGE = float(
    os.getenv("MIN_NIGHT_COVERAGE", "0.97")
)

MAX_OVERLAP_AGE_DAYS = int(
    os.getenv("MAX_OVERLAP_AGE_DAYS", "10")
)

STRICT_CALENDAR = (
    os.getenv("STRICT_TRADING_CALENDAR", "0") == "1"
)

MAX_BACKTEST_CALLS = int(
    os.getenv("MAX_BACKTEST_CALLS", "2500")
)

FINMIND_URL = (
    "https://api.finmindtrade.com/api/v4/data"
)

HORIZON = 5


# ============================================================
# 2. 共用工具與資料檢查
# ============================================================

def log(message):
    print(message, flush=True)


def write_json(name, data):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    (OUTPUT / name).write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def validate_ohlc(frame, label):
    """
    真實歷史 OHLC 使用嚴格檢查。
    AI 預測 OHLC 另外處理，不直接套用此函式。
    """

    if (
        frame.empty
        or not set(OHLC).issubset(frame.columns)
    ):
        raise ValueError(
            f"{label}: missing or empty OHLC"
        )

    arr = frame[OHLC].to_numpy(dtype=float)

    if (
        not np.isfinite(arr).all()
        or (arr <= 0).any()
    ):
        raise ValueError(
            f"{label}: invalid or nonpositive OHLC"
        )

    if (
        frame["high"]
        < frame[OHLC].max(axis=1)
    ).any():
        raise ValueError(
            f"{label}: high invalid"
        )

    if (
        frame["low"]
        > frame[OHLC].min(axis=1)
    ).any():
        raise ValueError(
            f"{label}: low invalid"
        )


# ============================================================
# 3. 下載加權指數
# ============================================================

def fetch_taiex():
    last_error = None

    for attempt in range(3):
        try:
            df = yf.download(
                "^TWII",
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
                threads=False
            )

            if df is None or df.empty:
                raise RuntimeError(
                    "Yahoo returned empty ^TWII"
                )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = (
                    df.columns.get_level_values(0)
                )

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close"
            })

            df = df[OHLC].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            df.index = pd.DatetimeIndex(df.index)

            if df.index.tz is not None:
                df.index = (
                    df.index
                    .tz_convert(TZ)
                    .tz_localize(None)
                )

            df.index = df.index.normalize()
            df = df.sort_index()

            if (
                df.index.has_duplicates
                or len(df) < 410
            ):
                raise RuntimeError(
                    "TAIEX duplicated dates "
                    "or insufficient history"
                )

            validate_ohlc(df, "TAIEX")

            log(
                f"[TAIEX] {len(df)} bars; "
                f"latest={df.index[-1].date()}"
            )

            return df

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(
                    3 * (attempt + 1)
                )

    raise RuntimeError(
        f"TAIEX failed: {last_error}"
    )


# ============================================================
# 4. FinMind 台指期下載
# ============================================================

def request_finmind(start, end):
    token = os.getenv(
        "FINMIND_TOKEN", ""
    ).strip()

    headers = (
        {"Authorization": f"Bearer {token}"}
        if token else {}
    )

    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": str(start),
        "end_date": str(end)
    }

    last = None

    for attempt in range(3):
        try:
            response = requests.get(
                FINMIND_URL,
                params=params,
                headers=headers,
                timeout=90
            )

            if response.status_code in (
                401, 402, 403, 429
            ):
                raise RuntimeError(
                    "FinMind authorization / rate limit: "
                    f"HTTP {response.status_code}"
                )

            response.raise_for_status()
            payload = response.json()

            if (
                payload.get("status") != 200
                or not isinstance(
                    payload.get("data"), list
                )
            ):
                raise RuntimeError(
                    "FinMind API error: "
                    f"{str(payload.get('msg'))[:180]}"
                )

            return pd.DataFrame(
                payload["data"]
            )

        except (
            requests.RequestException,
            ValueError
        ) as exc:
            last = exc

            if attempt < 2:
                time.sleep(
                    3 * (attempt + 1)
                )

    raise RuntimeError(
        f"FinMind request failed: {last}"
    )


def fetch_finmind():
    end = datetime.now(TZ).date()

    start = end - timedelta(days=1050)

    chunks = []
    current = start

    while current <= end:
        until = min(
            end,
            current + timedelta(days=179)
        )

        log(
            f"[TX] download {current}..{until}"
        )

        part = request_finmind(
            current, until
        )

        if not part.empty:
            chunks.append(part)

        current = until + timedelta(days=1)

    if not chunks:
        raise RuntimeError(
            "FinMind TX returned no data"
        )

    raw = pd.concat(
        chunks,
        ignore_index=True
    )

    log(f"[TX] raw rows={len(raw)}")

    return raw


# ============================================================
# 5. TX 日夜盤合併
# ============================================================

def make_tx_contract_bars(raw):
    """
    FinMind after_market 已使用歸屬交易日期。
    不再額外平移夜盤日期。
    """

    required = {
        "date",
        "futures_id",
        "contract_date",
        "trading_session",
        "open",
        "max",
        "min",
        "close",
        "volume"
    }

    if not required.issubset(raw.columns):
        raise RuntimeError(
            "TX missing columns: "
            f"{sorted(required - set(raw.columns))}"
        )

    df = raw.loc[
        raw["futures_id"].astype(str).str.upper() == "TX"
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.normalize()

    df["contract_date"] = (
        df["contract_date"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["contract_date"].str.fullmatch(
            r"20\d{4}"
        )
    ].copy()

    df = df[
        df["contract_date"]
        .str[-2:]
        .astype(int)
        .between(1, 12)
    ].copy()

    df = df[
        df["trading_session"].isin([
            "position",
            "after_market"
        ])
    ].copy()

    for key in (
        "open", "max", "min",
        "close", "volume"
    ):
        df[key] = pd.to_numeric(
            df[key],
            errors="coerce"
        )

    df = df.dropna(subset=[
        "date", "open", "max",
        "min", "close", "volume"
    ])

    df = df[
        (
            df[["open", "max", "min", "close"]] > 0
        ).all(axis=1)
        & (df["volume"] > 0)
        & (
            df["max"]
            >= df[["open", "min", "close"]].max(axis=1)
        )
        & (
            df["min"]
            <= df[["open", "max", "close"]].min(axis=1)
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No valid TX session rows"
        )

    keys = [
        "date",
        "contract_date",
        "trading_session"
    ]

    if df.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate TX session records"
        )

    rows = []

    for (date, contract), group in df.groupby(
        ["date", "contract_date"],
        sort=True
    ):
        day = group[
            group["trading_session"] == "position"
        ]

        night = group[
            group["trading_session"] == "after_market"
        ]

        if day.empty:
            continue

        d = day.iloc[0]

        n = (
            night.iloc[0]
            if not night.empty
            else None
        )

        sessions = (
            [d]
            if n is None
            else [n, d]
        )

        rows.append({
            "date": pd.Timestamp(date),
            "contract_date": str(contract),
            "open": float(
                sessions[0]["open"]
            ),
            "high": max(
                float(z["max"])
                for z in sessions
            ),
            "low": min(
                float(z["min"])
                for z in sessions
            ),
            "close": float(
                d["close"]
            ),
            "volume": sum(
                float(z["volume"])
                for z in sessions
            ),
            "has_night": n is not None
        })

    if not rows:
        raise RuntimeError(
            "No completed TX contract bars"
        )

    combined = pd.DataFrame(rows)

    combined = combined.sort_values([
        "date",
        "contract_date"
    ]).reset_index(drop=True)

    if combined.duplicated([
        "date",
        "contract_date"
    ]).any():
        raise RuntimeError(
            "TX daily contract rows duplicate"
        )

    validate_ohlc(
        combined,
        "TX contract bars"
    )

    return combined


# ============================================================
# 6. 近月合約與歷史轉倉
# ============================================================

def select_front_contract(combined):
    available = combined.loc[
        combined["contract_date"]
        >= combined["date"].dt.strftime("%Y%m")
    ]

    selected = (
        available
        .sort_values([
            "date",
            "contract_date"
        ])
        .drop_duplicates(
            "date",
            keep="first"
        )
    )

    if selected.empty:
        raise RuntimeError(
            "No TX front-month contract"
        )

    return selected.reset_index(drop=True)


def back_adjust_tx(selected, combined):
    """
    往前加減點數調整。

    計算轉倉價差時，只使用轉倉日及之前
    已知的新舊合約重疊收盤價。
    """

    adjusted = (
        selected
        .reset_index(drop=True)
        .copy()
    )

    quotes = combined.pivot(
        index="date",
        columns="contract_date",
        values="close"
    ).sort_index()

    roll_rows = []

    for i in range(1, len(adjusted)):
        old = str(
            adjusted.at[i - 1, "contract_date"]
        )

        new = str(
            adjusted.at[i, "contract_date"]
        )

        if old == new:
            continue

        roll_date = pd.Timestamp(
            adjusted.at[i, "date"]
        )

        if (
            new < old
            or old not in quotes.columns
            or new not in quotes.columns
        ):
            raise RuntimeError(
                f"Invalid TX roll: {old}->{new} "
                f"on {roll_date.date()}"
            )

        overlap = quotes.loc[
            :roll_date,
            [old, new]
        ].dropna()

        if (
            overlap.empty
            or (
                roll_date - overlap.index[-1]
            ).days > MAX_OVERLAP_AGE_DAYS
        ):
            raise RuntimeError(
                f"TX no recent overlap: "
                f"{old}->{new} on {roll_date.date()}"
            )

        reference_date = overlap.index[-1]

        gap = float(
            overlap.iloc[-1][new]
            - overlap.iloc[-1][old]
        )

        adjusted.loc[
            :i - 1, OHLC
        ] = adjusted.loc[
            :i - 1, OHLC
        ].add(gap)

        roll_rows.append({
            "roll_date": str(roll_date.date()),
            "reference_date": str(
                reference_date.date()
            ),
            "old_contract": old,
            "new_contract": new,
            "adjustment_points": gap
        })

    validate_ohlc(
        adjusted,
        "TX adjusted"
    )

    roll_log = pd.DataFrame(
        roll_rows,
        columns=[
            "roll_date",
            "reference_date",
            "old_contract",
            "new_contract",
            "adjustment_points"
        ]
    )

    return adjusted, roll_log


def tx_asof(
    combined,
    selected,
    end_date,
    count=400
):
    """
    只使用指定預測起點以前的 TX 行情。
    每個歷史預測日期重新建立連續合約。
    """

    end_date = pd.Timestamp(end_date)

    window = selected.loc[
        selected["date"] <= end_date
    ].tail(count)

    if len(window) < count:
        raise RuntimeError(
            f"TX {end_date.date()}: "
            f"{len(window)} bars, need {count}"
        )

    adjusted, rolls = back_adjust_tx(
        window,
        combined.loc[
            combined["date"] <= end_date
        ]
    )

    coverage = float(
        adjusted["has_night"].mean()
    )

    if coverage < MIN_NIGHT_COVERAGE:
        raise RuntimeError(
            f"TX night coverage {coverage:.1%} "
            f"< {MIN_NIGHT_COVERAGE:.1%}"
        )

    frame = (
        adjusted
        .set_index("date")[OHLC]
        .copy()
        .sort_index()
    )

    if frame.index.has_duplicates:
        raise RuntimeError(
            "Duplicate TX trading dates"
        )

    return frame, adjusted, rolls, coverage


# ============================================================
# 7. 日期檢查
# ============================================================

def check_dates(taiex, tx):
    a = pd.Timestamp(
        taiex.index[-1]
    )

    b = pd.Timestamp(
        tx.index[-1]
    )

    if a != b:
        raise RuntimeError(
            "TAIEX/TX cutoff mismatch: "
            f"{a.date()}, {b.date()}"
        )

    days = (
        datetime.now(TZ).date()
        - a.date()
    ).days

    if (
        days < 0
        or days > MAX_STALE_DAYS
    ):
        raise RuntimeError(
            f"Market data stale/future: "
            f"{days} calendar days"
        )


def future_market_dates(last_date):
    if not CALENDAR.exists():
        if STRICT_CALENDAR:
            raise RuntimeError(
                "Missing data/trading_calendar.csv"
            )

        log(
            "[CALENDAR] WARNING: weekdays only; "
            "Taiwan holidays ignored"
        )

        return (
            pd.bdate_range(
                pd.Timestamp(last_date)
                + timedelta(days=1),
                periods=HORIZON
            ),
            "weekdays_only"
        )

    cal = pd.read_csv(
        CALENDAR,
        dtype={"is_open": str}
    )

    if not {
        "date", "is_open"
    }.issubset(cal.columns):
        raise RuntimeError(
            "Calendar requires date,is_open"
        )

    cal["date"] = pd.to_datetime(
        cal["date"],
        errors="coerce"
    ).dt.normalize()

    if (
        cal["date"].isna().any()
        or cal["date"].duplicated().any()
    ):
        raise RuntimeError(
            "Calendar invalid dates or duplicates"
        )

    flags = (
        cal["is_open"]
        .str.lower()
        .str.strip()
    )

    if not flags.isin([
        "true", "false", "1", "0"
    ]).all():
        raise RuntimeError(
            "Calendar invalid is_open"
        )

    dates = cal.loc[
        (
            cal["date"] > pd.Timestamp(last_date)
        )
        & flags.isin(["true", "1"]),
        "date"
    ]

    dates = pd.DatetimeIndex(
        dates.sort_values().head(HORIZON)
    )

    if len(dates) < HORIZON:
        raise RuntimeError(
            "Calendar lacks 5 future trading days"
        )

    return dates, "csv_calendar"


# ============================================================
# 8. 多種子與情境設定
# ============================================================

def seed_everything(seed):
    random.seed(seed)

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(seed)


def experiment_seed(
    market,
    origin,
    replicate=0
):
    """
    相同市場、預測日期、種子編號，
    在 A/B/C/D 使用相同的起始亂數種子。
    """

    date_number = int(
        pd.Timestamp(origin).strftime("%Y%m%d")
    )

    market_offset = (
        0 if market == "TAIEX" else 1000
    )

    return (
        BASE_SEED
        + date_number
        + market_offset
        + replicate * 100003
    )


def selected_scenarios(value=None):
    value = (
        SELECTED
        if value is None
        else value
    )

    value = (
        value.replace(",", "")
        .replace(" ", "")
    )

    if (
        not value
        or any(
            c not in SCENARIO_CONFIG
            for c in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "KRONOS_SCENARIOS must contain "
            "unique A/B/C/D, e.g. AB or ABCD"
        )

    return [
        (key, *SCENARIO_CONFIG[key])
        for key in "ABCD"
        if key in value
    ]


# ============================================================
# 9. Kronos 模型推論與品質標記
# ============================================================

def predict(
    predictor,
    history,
    future_dates,
    sample_count,
    seed
):
    """
    保留 AI 原始預測 OHLC。

    若最高/最低價不合理，記錄 ohlc_valid=False。
    收盤價非有限正數則拒絕這次預測。
    """

    validate_ohlc(
        history,
        "Historical input"
    )

    if len(future_dates) != HORIZON:
        raise ValueError(
            "Forecast future dates must have length 5"
        )

    seed_everything(seed)

    x_dates = pd.Series(
        pd.DatetimeIndex(history.index)
    ).reset_index(drop=True)

    y_dates = pd.Series(
        pd.DatetimeIndex(future_dates)
    ).reset_index(drop=True)

    with torch.inference_mode():
        raw = predictor.predict(
            df=history[OHLC].reset_index(
                drop=True
            ),
            x_timestamp=x_dates,
            y_timestamp=y_dates,
            pred_len=HORIZON,
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False
        )

    if (
        not isinstance(raw, pd.DataFrame)
        or len(raw) != HORIZON
        or not set(OHLC).issubset(raw.columns)
    ):
        raise RuntimeError(
            "Kronos forecast unexpected shape "
            "or columns"
        )

    pred = (
        raw[OHLC]
        .copy()
        .reset_index(drop=True)
    )

    for col in OHLC:
        pred[col] = pd.to_numeric(
            pred[col],
            errors="coerce"
        )

    close = pred["close"].to_numpy(
        dtype=float
    )

    if (
        not np.isfinite(close).all()
        or (close <= 0).any()
    ):
        raise RuntimeError(
            "Kronos produced invalid predicted close"
        )

    price = pred[OHLC].to_numpy(
        dtype=float
    )

    finite_positive = (
        np.isfinite(price).all(axis=1)
        & (price > 0).all(axis=1)
    )

    tolerance = (
        pred["close"].abs() * 1e-6
    ).clip(lower=0.01)

    high_valid = (
        pred["high"] + tolerance
        >= pred[[
            "open", "low", "close"
        ]].max(axis=1)
    )

    low_valid = (
        pred["low"] - tolerance
        <= pred[[
            "open", "high", "close"
        ]].min(axis=1)
    )

    pred["ohlc_valid"] = (
        finite_positive
        & high_valid.to_numpy()
        & low_valid.to_numpy()
    )

    invalid_count = int(
        (~pred["ohlc_valid"]).sum()
    )

    if invalid_count:
        log(
            "[QUALITY] "
            f"origin={history.index[-1].date()}, "
            f"samples={sample_count}, "
            f"seed={seed}, "
            f"invalid OHLC={invalid_count}/5 "
            "(raw prices retained)"
        )

    pred.index = pd.DatetimeIndex(
        future_dates
    )

    pred.index.name = "date"

    reference = float(
        history["close"].iloc[-1]
    )

    pred["return_pct"] = (
        (
            pred["close"] / reference
        ) - 1
    ) * 100

    return pred


# ============================================================
# 10. 歷史極端值參考
# ============================================================

def historical_reference(taiex, tx):
    aligned = pd.concat(
        {
            "TAIEX": taiex["close"],
            "TX": tx["close"]
        },
        axis=1
    ).dropna()

    records = []

    for h in (1, 3, 5):
        returns = aligned.pct_change(
            h,
            fill_method=None
        ).dropna() * 100

        records.append({
            "day": h,
            "historical_abs_gap_p99_pp": float(
                (
                    returns["TX"]
                    - returns["TAIEX"]
                ).abs().quantile(.99)
            ),
            "historical_abs_tx_return_p99_pct": float(
                returns["TX"].abs().quantile(.99)
            )
        })

    ref = pd.DataFrame(records)

    ref.to_csv(
        OUTPUT / "historical_reference.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return ref


# ============================================================
# 11. 每日預測圖表
# ============================================================

def plot_comparison(summary, market):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for label, part in summary.groupby(
        "scenario"
    ):
        ax.plot(
            part["day"],
            part[f"{market}_return_pct"],
            marker="o",
            label=label
        )

    ax.axhline(
        0,
        linewidth=0.7,
        color="gray"
    )

    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Future trading days")
    ax.set_ylabel("Return (%)")

    ax.set_title(
        f"Kronos-base {market}: scenario comparison"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{market}_scenario_comparison.png",
        dpi=145
    )

    plt.close(fig)


def plot_single_market(
    history,
    pred,
    market,
    scenario
):
    recent = history["close"].tail(60)

    anchor = pd.Series(
        [float(recent.iloc[-1])],
        index=[recent.index[-1]]
    )

    path = pd.concat([
        anchor,
        pred["close"]
    ])

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        recent.index,
        recent.to_numpy(),
        label="History"
    )

    ax.plot(
        path.index,
        path.to_numpy(),
        "--o",
        label=f"Forecast {scenario}"
    )

    ax.set_title(
        f"{market} Kronos-base ({scenario})"
    )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{market}_forecast.png",
        dpi=145
    )

    plt.close(fig)


# ============================================================
# 12. 每日預測模式
# ============================================================

def run_experiments(
    predictor,
    taiex,
    tx,
    future_dates,
    calendar_mode,
    scenarios=None
):
    scenarios = (
        SCENARIOS
        if scenarios is None
        else scenarios
    )

    reference = historical_reference(
        taiex, tx
    ).set_index("day")

    results = []
    rows = []

    selected_labels = [
        item[0]
        for item in scenarios
    ]

    for label, lookback, samples in scenarios:
        forecasts = {}

        for market, frame in (
            ("TAIEX", taiex),
            ("TX", tx)
        ):
            history = frame.tail(
                lookback
            )

            if len(history) < lookback:
                raise RuntimeError(
                    f"{market}: insufficient "
                    f"{lookback} daily bars"
                )

            seed = experiment_seed(
                market,
                frame.index[-1],
                replicate=0
            )

            log(
                f"[LIVE] {label} {market} "
                f"lookback={lookback}, "
                f"samples={samples}, seed={seed}"
            )

            pred = predict(
                predictor,
                history,
                future_dates,
                samples,
                seed
            )

            pred.to_csv(
                OUTPUT / f"{market}_{label}_forecast.csv",
                encoding="utf-8-sig"
            )

            # 相容原有輸出檔案
            if (
                label == "D"
                or (
                    label == "B"
                    and "D" not in selected_labels
                )
            ):
                pred.to_csv(
                    OUTPUT / f"{market}_forecast.csv",
                    encoding="utf-8-sig"
                )

                plot_single_market(
                    frame,
                    pred,
                    market,
                    label
                )

            forecasts[market] = pred

            results.append({
                "market": market,
                "scenario": label,
                "lookback": lookback,
                "samples": samples,
                "seed": seed,
                "day_1_pct": float(
                    pred["return_pct"].iloc[0]
                ),
                "day_3_pct": float(
                    pred["return_pct"].iloc[2]
                ),
                "day_5_pct": float(
                    pred["return_pct"].iloc[4]
                ),
                "path_ohlc_valid": bool(
                    pred["ohlc_valid"].all()
                )
            })

        for h in (1, 3, 5):
            a = float(
                forecasts["TAIEX"]["return_pct"].iloc[h - 1]
            )

            b = float(
                forecasts["TX"]["return_pct"].iloc[h - 1]
            )

            ref = reference.loc[h]
            gap = b - a

            review = (
                abs(gap)
                > ref["historical_abs_gap_p99_pp"]
                or abs(b)
                > ref[
                    "historical_abs_tx_return_p99_pct"
                ]
            )

            rows.append({
                "scenario": label,
                "lookback": lookback,
                "sample_count": samples,
                "day": h,
                "forecast_date": str(
                    future_dates[h - 1].date()
                ),
                "TAIEX_return_pct": a,
                "TX_return_pct": b,
                "return_gap_pp": gap,
                "TAIEX_ohlc_valid": bool(
                    forecasts["TAIEX"][
                        "ohlc_valid"
                    ].iloc[h - 1]
                ),
                "TX_ohlc_valid": bool(
                    forecasts["TX"][
                        "ohlc_valid"
                    ].iloc[h - 1]
                ),
                "review_status": (
                    "REVIEW" if review else "OK"
                )
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT / "experiment_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for market in ("TAIEX", "TX"):
        plot_comparison(
            summary,
            market
        )

    report = [
        "# Kronos-base V3.2 每日參數敏感度比較",
        "",
        (
            f"資料截止日：{taiex.index[-1].date()}"
            f"；日期模式：{calendar_mode}"
        ),
        "",
        "REVIEW 為歷史極端值標記，"
        "並不代表預測準確或具有交易優勢。",
        "",
        "| 情境 | 日數 | TAIEX % | TX % | 差距 pp | 狀態 |",
        "|---|---:|---:|---:|---:|---|"
    ]

    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.scenario} | {row.day} | "
            f"{row.TAIEX_return_pct:+.2f} | "
            f"{row.TX_return_pct:+.2f} | "
            f"{row.return_gap_pp:+.2f} | "
            f"{row.review_status} |"
        )

    report += [
        "",
        "## 參數敏感度",
        "固定亂數種子有助實驗重現，"
        "但不能取代歷史回測。"
    ]

    (
        OUTPUT / "experiment_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    return results, {
        "scenario_count": len(scenarios),
        "market_forecast_runs": len(results),
        "comparisons": len(summary),
        "review_rows": int(
            (
                summary["review_status"] == "REVIEW"
            ).sum()
        )
    }


# ============================================================
# 13. TX 歷史已實現報酬
# ============================================================

def realized_tx_return(
    origin,
    target,
    selected,
    combined
):
    """
    只用於事後評分，非實際可成交損益。

    模型輸入在 origin 截止；
    評分才使用 target 日期的已實現價格。
    """

    obs = selected[
        selected["date"] <= target
    ].tail(400)

    adj, _ = back_adjust_tx(
        obs,
        combined[
            combined["date"] <= target
        ]
    )

    close = (
        adj.set_index("date")["close"]
    )

    return (
        float(close.loc[target])
        / float(close.loc[origin])
        - 1
    ) * 100


def momentum_baseline(
    history,
    horizon
):
    """
    五日動能基準，按預測長度等比延伸。
    """

    r5 = float(
        history["close"].iloc[-1]
        / history["close"].iloc[-6]
        - 1
    )

    return (
        (1 + r5) ** (horizon / 5)
        - 1
    ) * 100


# ============================================================
# 14. 多種子回測統計
# ============================================================

def summarize_backtest(detail):
    """
    先將同一 origin 的成功種子預測取平均，
    然後每個 origin 只計算一次準確率與 MAE。

    不把三個種子視為三個獨立交易日。
    """

    valid = detail[
        detail["prediction_ok"]
        & np.isfinite(
            detail["pred_return_pct"]
        )
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "All backtest predictions failed; "
            "see backtest_detail.csv"
        )

    groups = [
        "origin",
        "market",
        "scenario",
        "horizon"
    ]

    quality = (
        detail.groupby(groups)
        .agg(
            attempts=(
                "prediction_ok", "size"
            ),
            valid_count=(
                "prediction_ok", "sum"
            ),
            ohlc_valid_rate=(
                "ohlc_valid", "mean"
            ),
            full_path_valid_rate=(
                "path_ohlc_valid", "mean"
            )
        )
        .reset_index()
    )

    origins = (
        valid.groupby(groups)
        .agg(
            pred_return_pct=(
                "pred_return_pct", "mean"
            ),
            pred_std_across_seeds_pp=(
                "pred_return_pct", "std"
            ),
            actual_return_pct=(
                "actual_return_pct", "first"
            ),
            flat_return_pct=(
                "flat_return_pct", "first"
            ),
            momentum_return_pct=(
                "momentum_return_pct", "first"
            )
        )
        .reset_index()
    )

    origins = origins.merge(
        quality,
        on=groups,
        how="left"
    )

    origins[
        "pred_std_across_seeds_pp"
    ] = origins[
        "pred_std_across_seeds_pp"
    ].fillna(0)

    origins.to_csv(
        OUTPUT / "backtest_origin_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rows = []

    for (
        market, scenario, h
    ), g in origins.groupby([
        "market",
        "scenario",
        "horizon"
    ]):
        error = (
            g["pred_return_pct"]
            - g["actual_return_pct"]
        )

        flat_error = (
            g["actual_return_pct"]
            - g["flat_return_pct"]
        )

        mom_error = (
            g["actual_return_pct"]
            - g["momentum_return_pct"]
        )

        mae = float(
            error.abs().mean()
        )

        flat_mae = float(
            flat_error.abs().mean()
        )

        mom_mae = float(
            mom_error.abs().mean()
        )

        rows.append({
            "market": market,
            "scenario": scenario,
            "horizon": int(h),
            "n_origins": int(len(g)),
            "seed_count_requested": SEED_COUNT,
            "avg_valid_seeds_per_origin": round(
                float(g["valid_count"].mean()),
                2
            ),
            "forecast_success_rate": round(
                float(
                    g["valid_count"].sum()
                    / g["attempts"].sum()
                ),
                5
            ),
            "ohlc_valid_rate": round(
                float(g["ohlc_valid_rate"].mean()),
                5
            ),
            "path_ohlc_valid_rate": round(
                float(
                    g["full_path_valid_rate"].mean()
                ),
                5
            ),
            "direction_accuracy": float(
                (
                    np.sign(g["pred_return_pct"])
                    == np.sign(
                        g["actual_return_pct"]
                    )
                ).mean()
            ),
            "mae_pp": mae,
            "rmse_pp": float(
                np.sqrt(
                    np.square(error).mean()
                )
            ),
            "flat_mae_pp": flat_mae,
            "momentum_mae_pp": mom_mae,
            "beats_flat_mae": mae < flat_mae,
            "beats_momentum_mae": mae < mom_mae,
            "mean_seed_std_pp": float(
                g[
                    "pred_std_across_seeds_pp"
                ].mean()
            ),
            "max_seed_std_pp": float(
                g[
                    "pred_std_across_seeds_pp"
                ].max()
            )
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values([
        "market",
        "horizon",
        "scenario"
    ])

    summary.to_csv(
        OUTPUT / "backtest_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 分別評估各個亂數種子的表現
    seed_rows = []

    for (
        market, scenario, h, replicate
    ), g in valid.groupby([
        "market",
        "scenario",
        "horizon",
        "seed_index"
    ]):
        error = (
            g["pred_return_pct"]
            - g["actual_return_pct"]
        )

        seed_rows.append({
            "market": market,
            "scenario": scenario,
            "horizon": int(h),
            "seed_index": int(replicate),
            "n_origins": len(g),
            "direction_accuracy": float(
                (
                    np.sign(g["pred_return_pct"])
                    == np.sign(
                        g["actual_return_pct"]
                    )
                ).mean()
            ),
            "mae_pp": float(
                error.abs().mean()
            ),
            "rmse_pp": float(
                np.sqrt(
                    np.square(error).mean()
                )
            )
        })

    pd.DataFrame(
        seed_rows
    ).to_csv(
        OUTPUT / "backtest_seed_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return origins, summary


# ============================================================
# 15. 回測圖表及報告
# ============================================================

def plot_backtest(summary):
    for market in ("TAIEX", "TX"):
        data = summary[
            summary["market"] == market
        ]

        if data.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        for scenario, part in data.groupby(
            "scenario"
        ):
            ax.plot(
                part["horizon"],
                part["mae_pp"],
                "-o",
                label=scenario
            )

        flat = (
            data.groupby("horizon")[
                "flat_mae_pp"
            ].first()
        )

        ax.plot(
            flat.index,
            flat.to_numpy(),
            "--",
            label="Flat baseline"
        )

        ax.set_xticks([1, 3, 5])

        ax.set_ylabel(
            "Return MAE (percentage points)"
        )

        ax.set_xlabel(
            "Trading days ahead"
        )

        ax.set_title(
            f"{market} V3.2 rolling MAE"
        )

        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            OUTPUT / f"backtest_{market.lower()}_mae.png",
            dpi=140
        )

        plt.close(fig)


def create_backtest_report(
    summary,
    days
):
    lines = [
        "# Kronos-base V3.2 多種子歷史滾動回測",
        "",
        (
            f"歷史預測起點：{days} 日；"
            f"種子數：{SEED_COUNT}；"
            f"設定：{SELECTED}"
        ),
        "",
        "同一歷史日期先將有效種子的預測取平均，"
        "再計算 MAE，避免將多種子視為獨立觀測。",
        "",
        "注意：模型可能曾經使用部分回測日期的"
        "資料進行預訓練，不能直接宣稱是訓練外樣本。",
        "",
        "TX 使用後調整連續合約評估報酬，"
        "不是實際可成交損益。",
        "",
        "| 市場 | 情境 | 日數 | 有效起點 | "
        "方向正確率 | MAE(pp) | RMSE(pp) | "
        "Flat MAE | 動能 MAE | 種子標準差(pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for r in summary.itertuples(
        index=False
    ):
        lines.append(
            f"| {r.market} | "
            f"{r.scenario} | "
            f"{r.horizon} | "
            f"{r.n_origins} | "
            f"{r.direction_accuracy:.1%} | "
            f"{r.mae_pp:.3f} | "
            f"{r.rmse_pp:.3f} | "
            f"{r.flat_mae_pp:.3f} | "
            f"{r.momentum_mae_pp:.3f} | "
            f"{r.mean_seed_std_pp:.3f} |"
        )

    lines += [
        "",
        "OHLC 異常及失敗紀錄請另查看 "
        "`backtest_detail.csv`。",
        "",
        "五日報酬相互重疊，20 日樣本偏少，"
        "不能把方向正確率直接視為長期交易勝率。"
    ]

    (
        OUTPUT / "backtest_report.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


# ============================================================
# 16. 歷史滾動回測核心
# ============================================================

def run_backtest(
    predictor,
    taiex,
    combined,
    days,
    scenarios=None
):
    scenarios = (
        selected_scenarios()
        if scenarios is None
        else scenarios
    )

    selected = select_front_contract(
        combined
    )

    common = taiex.index.intersection(
        pd.DatetimeIndex(
            selected["date"]
        )
    ).sort_values()

    if len(common) < 400 + HORIZON:
        raise RuntimeError(
            f"Only {len(common)} TAIEX/TX "
            "common dates"
        )

    candidates = common[
        399:-HORIZON
    ]

    if len(candidates) < days:
        raise RuntimeError(
            f"Only {len(candidates)} historical "
            f"origins, need {days}"
        )

    origins = candidates[-days:]

    expected_calls = (
        len(origins)
        * len(scenarios)
        * 2
        * SEED_COUNT
    )

    if expected_calls > MAX_BACKTEST_CALLS:
        raise RuntimeError(
            f"{expected_calls} inference calls "
            f"exceed MAX_BACKTEST_CALLS="
            f"{MAX_BACKTEST_CALLS}"
        )

    log(
        f"[BACKTEST V3.2] origins={len(origins)}, "
        f"scenarios={''.join(s[0] for s in scenarios)}, "
        f"seeds={SEED_COUNT}, "
        f"model_calls={expected_calls}"
    )

    rows = []

    for index, origin in enumerate(
        origins, 1
    ):
        started = time.monotonic()

        location = common.get_loc(origin)

        future_dates = common[
            location + 1:
            location + 1 + HORIZON
        ]

        # 只使用此預測日期以前的期貨行情
        tx_hist, _, _, coverage = tx_asof(
            combined,
            selected,
            origin,
            400
        )

        hist_map = {
            "TAIEX": taiex.loc[
                :origin
            ].tail(400),
            "TX": tx_hist
        }

        realized = {}

        # 實際結果只用於事後評分
        for market in ("TAIEX", "TX"):
            realized[market] = {}

            for h in (1, 3, 5):
                target = future_dates[
                    h - 1
                ]

                if market == "TAIEX":
                    actual = (
                        float(
                            taiex.at[target, "close"]
                        )
                        / float(
                            taiex.at[origin, "close"]
                        )
                        - 1
                    ) * 100

                else:
                    actual = realized_tx_return(
                        origin,
                        target,
                        selected,
                        combined
                    )

                realized[market][h] = actual

        # 依情境、市場、種子依序推論
        for label, count, samples in scenarios:
            for market in ("TAIEX", "TX"):
                hist = hist_map[
                    market
                ].tail(count)

                if len(hist) != count:
                    raise RuntimeError(
                        f"Not enough {market} bars "
                        f"at {origin.date()}"
                    )

                for replicate in range(
                    SEED_COUNT
                ):
                    seed = experiment_seed(
                        market,
                        origin,
                        replicate
                    )

                    forecast = None
                    problem = ""

                    try:
                        forecast = predict(
                            predictor,
                            hist,
                            future_dates,
                            samples,
                            seed
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        FloatingPointError
                    ) as exc:
                        problem = (
                            f"{type(exc).__name__}: "
                            f"{str(exc)[:200]}"
                        )

                        log(
                            f"[PREDICT FAILED] "
                            f"{origin.date()} "
                            f"{market}/{label}/"
                            f"seed{replicate}: {problem}"
                        )

                    for h in (1, 3, 5):
                        actual = realized[
                            market
                        ][h]

                        rows.append({
                            "origin": str(
                                origin.date()
                            ),
                            "target_date": str(
                                future_dates[
                                    h - 1
                                ].date()
                            ),
                            "market": market,
                            "scenario": label,
                            "lookback": count,
                            "samples": samples,
                            "seed_index": replicate,
                            "random_seed": seed,
                            "horizon": h,
                            "prediction_ok": (
                                forecast is not None
                            ),
                            "pred_return_pct": (
                                float(
                                    forecast[
                                        "return_pct"
                                    ].iloc[h - 1]
                                )
                                if forecast is not None
                                else np.nan
                            ),
                            "actual_return_pct": actual,
                            "flat_return_pct": 0.0,
                            "momentum_return_pct":
                                momentum_baseline(
                                    hist, h
                                ),
                            "ohlc_valid": (
                                bool(
                                    forecast[
                                        "ohlc_valid"
                                    ].iloc[h - 1]
                                )
                                if forecast is not None
                                else False
                            ),
                            "path_ohlc_valid": (
                                bool(
                                    forecast[
                                        "ohlc_valid"
                                    ].all()
                                )
                                if forecast is not None
                                else False
                            ),
                            "error": problem
                        })

        # 每五個預測起點保存一次中間結果
        if (
            index % 5 == 0
            or index == len(origins)
        ):
            pd.DataFrame(rows).to_csv(
                OUTPUT / "backtest_detail.csv",
                index=False,
                encoding="utf-8-sig"
            )

        log(
            f"[BACKTEST] {index}/{len(origins)} "
            f"origin={origin.date()} "
            f"elapsed="
            f"{time.monotonic() - started:.1f}s"
        )

    detail = pd.DataFrame(rows)

    origins_summary, summary = (
        summarize_backtest(detail)
    )

    plot_backtest(summary)

    create_backtest_report(
        summary,
        len(origins)
    )

    return {
        "origins": len(origins),
        "model_calls_planned": expected_calls,
        "forecast_rows": len(detail),
        "seed_count": SEED_COUNT,
        "origin_aggregate_rows": len(
            origins_summary
        ),
        "summary_rows": len(summary),
        "first_origin": str(
            origins[0].date()
        ),
        "last_origin": str(
            origins[-1].date()
        )
    }


# ============================================================
# 17. 主程式
# ============================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    start = time.monotonic()

    status = {
        "success": False,
        "version": "V3.2",
        "mode": MODE,
        "model": MODEL_NAME,
        "created_at_taipei":
            datetime.now(TZ).isoformat(),
        "config": {
            "backtest_days": BACKTEST_DAYS,
            "seed_count": SEED_COUNT,
            "base_seed": BASE_SEED,
            "selected_scenarios": SELECTED
        }
    }

    try:
        if (
            MODE not in ("live", "backtest")
            or not (1 <= BACKTEST_DAYS <= 100)
        ):
            raise ValueError(
                "KRONOS_MODE=live|backtest "
                "and BACKTEST_DAYS=1..100 required"
            )

        if SEED_COUNT not in (1, 3, 5):
            raise ValueError(
                "KRONOS_SEED_COUNT must be 1, 3, or 5"
            )

        scenarios = selected_scenarios()

        if not (
            0 <= MIN_NIGHT_COVERAGE <= 1
        ):
            raise ValueError(
                "MIN_NIGHT_COVERAGE must be "
                "between 0 and 1"
            )

        log(
            f"[V3.2] mode={MODE}; "
            f"scenarios={SELECTED}; "
            f"seeds={SEED_COUNT}"
        )

        # 下載最新行情
        taiex = fetch_taiex()
        raw = fetch_finmind()

        combined = make_tx_contract_bars(
            raw
        )

        combined.to_csv(
            OUTPUT / "tx_contract_bars.csv",
            index=False,
            encoding="utf-8-sig"
        )

        if MODE == "live":
            selected = select_front_contract(
                combined
            )

            tx, adjusted, rolls, coverage = tx_asof(
                combined,
                selected,
                selected["date"].iloc[-1],
                400
            )

            check_dates(
                taiex,
                tx
            )

            taiex.tail(400).to_csv(
                OUTPUT / "taiex_history.csv",
                encoding="utf-8-sig"
            )

            adjusted.to_csv(
                OUTPUT / "tx_continuous_adjusted.csv",
                index=False,
                encoding="utf-8-sig"
            )

            rolls.to_csv(
                OUTPUT / "tx_roll_log.csv",
                index=False,
                encoding="utf-8-sig"
            )

            future_dates, calendar_mode = (
                future_market_dates(
                    taiex.index[-1]
                )
            )

            status.update({
                "latest_data_date": str(
                    taiex.index[-1].date()
                ),
                "night_coverage": coverage,
                "calendar_mode": calendar_mode
            })

        # Kronos-base 模型只載入一次
        log(
            f"[MODEL] loading {MODEL_NAME} on CPU"
        )

        tokenizer = KronosTokenizer.from_pretrained(
            TOKENIZER_NAME
        )

        model = Kronos.from_pretrained(
            MODEL_NAME
        )

        tokenizer.eval()
        model.eval()

        predictor = KronosPredictor(
            model,
            tokenizer,
            device="cpu",
            max_context=512
        )

        # 每日預測
        if MODE == "live":
            (
                status["results"],
                status["diagnostics"]
            ) = run_experiments(
                predictor,
                taiex.tail(400),
                tx,
                future_dates,
                calendar_mode,
                scenarios
            )

        # 歷史回測
        else:
            status["backtest"] = run_backtest(
                predictor,
                taiex,
                combined,
                BACKTEST_DAYS,
                scenarios
            )

        status["success"] = True

        log(
            "[V3.2 SUCCESS] Completed"
        )

    except Exception as exc:
        status.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(
                limit=12
            )
        })

        log(
            f"[V3.2 FAILED] "
            f"{type(exc).__name__}: {exc}"
        )

        raise

    finally:
        status["elapsed_seconds"] = round(
            time.monotonic() - start,
            2
        )

        write_json(
            "status.json",
            status
        )


if __name__ == "__main__":
    main()
