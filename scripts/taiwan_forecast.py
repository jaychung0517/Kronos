
"""Kronos-base V3: TAIEX and TX day+night; live scenarios and rolling backtest.

Run from Kronos repository root:
    python -m scripts.taiwan_forecast

Uses Yahoo ^TWII and FinMind TaiwanFuturesDaily (TX).
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

MODEL_NAME = os.getenv(
    "KRONOS_MODEL",
    "NeoQuasar/Kronos-base"
)

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

MODE = os.getenv(
    "KRONOS_MODE", "live"
).lower().strip()

BACKTEST_DAYS = int(
    os.getenv("BACKTEST_DAYS", "100")
)

BASE_SEED = int(
    os.getenv("KRONOS_SEED", "20261104")
)

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

HORIZON = 5

OHLC = ["open", "high", "low", "close"]

SCENARIOS = [
    ("A", 200, 1),
    ("B", 200, 5),
    ("C", 400, 1),
    ("D", 400, 5),
]

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# ============================================================
# 2. 共用工具與行情檢查
# ============================================================

def log(message):
    print(message, flush=True)


def save_json(name, payload):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    (OUTPUT / name).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def validate_ohlc(frame, label):
    if frame.empty or not set(OHLC).issubset(frame.columns):
        raise ValueError(
            f"{label}: empty or missing OHLC"
        )

    x = frame[OHLC].to_numpy(dtype=float)

    if not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError(
            f"{label}: nonpositive or nonfinite prices"
        )

    if (frame.high < frame[OHLC].max(axis=1)).any():
        raise ValueError(
            f"{label}: high is invalid"
        )

    if (frame.low > frame[OHLC].min(axis=1)).any():
        raise ValueError(
            f"{label}: low is invalid"
        )


# ============================================================
# 3. 下載加權指數
# ============================================================

def fetch_taiex():
    error = None

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
                    "Yahoo ^TWII returned no data"
                )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

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
                    df.index.tz_convert(TZ).tz_localize(None)
                )

            df.index = df.index.normalize()
            df = df.sort_index()

            if df.index.has_duplicates or len(df) < 510:
                raise RuntimeError(
                    "TAIEX has duplicates or insufficient history"
                )

            validate_ohlc(df, "TAIEX")

            log(
                f"[TAIEX] {len(df)} bars "
                f"through {df.index[-1].date()}"
            )

            return df

        except Exception as exc:
            error = exc

            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    raise RuntimeError(
        f"TAIEX download failed: {error}"
    )


# ============================================================
# 4. FinMind 台指期資料下載
# ============================================================

def request_finmind(start, end):
    headers = {}

    token = os.getenv(
        "FINMIND_TOKEN", ""
    ).strip()

    if token:
        headers["Authorization"] = f"Bearer {token}"

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
                    "FinMind authorization/quota "
                    f"HTTP {response.status_code}"
                )

            response.raise_for_status()
            payload = response.json()

            if (
                payload.get("status") != 200
                or not isinstance(payload.get("data"), list)
            ):
                raise RuntimeError(
                    "FinMind API error: "
                    f"{str(payload.get('msg'))[:200]}"
                )

            return pd.DataFrame(payload["data"])

        except (
            requests.RequestException,
            ValueError
        ) as exc:
            last = exc

            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    raise RuntimeError(
        f"FinMind download failed: {last}"
    )


def fetch_finmind():
    end = datetime.now(TZ).date()
    start = end - timedelta(days=1050)

    parts = []
    date = start

    while date <= end:
        until = min(
            end,
            date + timedelta(days=179)
        )

        log(
            f"[TX] Fetching {date}..{until}"
        )

        chunk = request_finmind(date, until)

        if not chunk.empty:
            parts.append(chunk)

        date = until + timedelta(days=1)

    if not parts:
        raise RuntimeError(
            "FinMind returned no TX rows"
        )

    df = pd.concat(
        parts,
        ignore_index=True
    )

    log(f"[TX] raw rows={len(df)}")

    return df


# ============================================================
# 5. TX 日夜盤合併
# ============================================================

def make_tx_contract_bars(raw):
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
            "FinMind TX missing "
            f"{sorted(required - set(raw.columns))}"
        )

    df = raw[
        raw.futures_id.astype(str).str.upper() == "TX"
    ].copy()

    df["date"] = pd.to_datetime(
        df.date,
        errors="coerce"
    ).dt.normalize()

    df["contract_date"] = (
        df.contract_date.astype(str).str.strip()
    )

    df = df[
        df.contract_date.str.fullmatch(r"20\d{4}")
    ].copy()

    df = df[
        df.contract_date.str[-2:].astype(int).between(1, 12)
    ].copy()

    df = df[
        df.trading_session.isin([
            "position",
            "after_market"
        ])
    ].copy()

    for col in (
        "open", "max", "min", "close", "volume"
    ):
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=[
        "date", "open", "max",
        "min", "close", "volume"
    ])

    df = df[
        (df[["open", "max", "min", "close"]] > 0).all(axis=1)
        & (df.volume > 0)
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
            "No valid TX sessions"
        )

    keys = [
        "date",
        "contract_date",
        "trading_session"
    ]

    if df.duplicated(keys).any():
        raise RuntimeError(
            "Duplicated TX date/contract/session; "
            "inspect raw source"
        )

    rows = []

    # FinMind 夜盤已使用所歸屬的期貨交易日期。
    # 不要自行再將夜盤日期加一天或減一天。

    for (date, contract), group in df.groupby(
        ["date", "contract_date"]
    ):
        day = group[
            group.trading_session == "position"
        ]

        night = group[
            group.trading_session == "after_market"
        ]

        if day.empty:
            continue

        day = day.iloc[0]

        have_night = not night.empty

        sessions = [day]

        if have_night:
            sessions.insert(
                0,
                night.iloc[0]
            )

        rows.append({
            "date": pd.Timestamp(date),
            "contract_date": contract,
            "open": float(sessions[0]["open"]),
            "high": float(
                max(x["max"] for x in sessions)
            ),
            "low": float(
                min(x["min"] for x in sessions)
            ),
            "close": float(day["close"]),
            "volume": float(
                sum(x["volume"] for x in sessions)
            ),
            "has_night": bool(have_night)
        })

    combined = pd.DataFrame(rows)

    if combined.empty:
        raise RuntimeError(
            "TX has no completed day sessions"
        )

    combined = combined.sort_values([
        "date", "contract_date"
    ]).reset_index(drop=True)

    if combined.duplicated([
        "date", "contract_date"
    ]).any():
        raise RuntimeError(
            "TX duplicated combined contracts"
        )

    validate_ohlc(
        combined,
        "TX combined"
    )

    return combined


# ============================================================
# 6. 近月合約選擇與期貨換月調整
# ============================================================

def select_front_contract(combined):
    df = combined[
        combined.contract_date
        >= combined.date.dt.strftime("%Y%m")
    ]

    return (
        df.sort_values([
            "date",
            "contract_date"
        ])
        .drop_duplicates(
            "date",
            keep="first"
        )
        .reset_index(drop=True)
    )


def back_adjust_tx(selected, combined):
    adjusted = (
        selected.copy().reset_index(drop=True)
    )

    quotes = combined.pivot(
        index="date",
        columns="contract_date",
        values="close"
    ).sort_index()

    logs = []

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
            or old not in quotes
            or new not in quotes
        ):
            raise RuntimeError(
                f"Invalid TX roll "
                f"{roll_date.date()}: {old}->{new}"
            )

        both = quotes.loc[
            :roll_date,
            [old, new]
        ].dropna()

        if (
            both.empty
            or (
                roll_date - both.index[-1]
            ).days > MAX_OVERLAP_AGE_DAYS
        ):
            raise RuntimeError(
                "Missing recent overlap for "
                f"TX roll {old}->{new}"
            )

        gap = float(
            both.iloc[-1][new]
            - both.iloc[-1][old]
        )

        adjusted.loc[
            :i - 1,
            OHLC
        ] = adjusted.loc[
            :i - 1,
            OHLC
        ].add(gap)

        logs.append({
            "roll_date": str(roll_date.date()),
            "reference_date": str(
                both.index[-1].date()
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
        logs,
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
    selected_all,
    end_date,
    count=400
):
    """
    僅使用指定歷史日期以前的 TX 行情。

    用於歷史回測時，避免未來換月資訊進入模型。
    """

    past = selected_all[
        selected_all.date <= pd.Timestamp(end_date)
    ].tail(count).copy()

    if len(past) < count:
        raise RuntimeError(
            f"Only {len(past)} TX bars available; "
            f"need {count}"
        )

    quotes = combined[
        combined.date <= pd.Timestamp(end_date)
    ]

    adjusted, rolls = back_adjust_tx(
        past,
        quotes
    )

    coverage = float(
        adjusted.has_night.mean()
    )

    if coverage < MIN_NIGHT_COVERAGE:
        raise RuntimeError(
            f"TX night coverage {coverage:.1%} "
            f"below {MIN_NIGHT_COVERAGE:.1%}"
        )

    frame = adjusted.set_index(
        "date"
    )[OHLC].sort_index()

    if frame.index.has_duplicates:
        raise RuntimeError(
            "Duplicate TX dates"
        )

    return frame, adjusted, rolls, coverage


# ============================================================
# 7. 日期檢查
# ============================================================

def check_dates(taiex, tx):
    last_a = taiex.index[-1]
    last_b = tx.index[-1]

    if last_a != last_b:
        raise RuntimeError(
            "Market dates differ: "
            f"{last_a.date()} vs {last_b.date()}"
        )

    days = (
        datetime.now(TZ).date()
        - last_a.date()
    ).days

    if days < 0 or days > MAX_STALE_DAYS:
        raise RuntimeError(
            f"Market data stale/future: "
            f"{last_a.date()}, age={days}"
        )


def future_market_dates(last_date):
    if not CALENDAR.exists():
        if STRICT_CALENDAR:
            raise RuntimeError(
                "Missing data/trading_calendar.csv"
            )

        log(
            "[CALENDAR] WARNING: weekdays only; "
            "Taiwan holidays not removed"
        )

        dates = pd.bdate_range(
            pd.Timestamp(last_date)
            + timedelta(days=1),
            periods=HORIZON
        )

        return dates, "weekdays_only"

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

    cal.date = pd.to_datetime(
        cal.date,
        errors="coerce"
    )

    if (
        cal.date.isna().any()
        or cal.date.duplicated().any()
    ):
        raise RuntimeError(
            "Invalid or duplicated calendar dates"
        )

    flags = (
        cal.is_open.str.lower().str.strip()
    )

    if not flags.isin([
        "true", "false", "1", "0"
    ]).all():
        raise RuntimeError(
            "Calendar is_open must be true/false"
        )

    dates = cal.loc[
        (cal.date > pd.Timestamp(last_date))
        & flags.isin(["true", "1"]),
        "date"
    ]

    dates = pd.DatetimeIndex(
        dates.sort_values().head(HORIZON)
    )

    if len(dates) != HORIZON:
        raise RuntimeError(
            "Calendar insufficient for "
            "five future trading days"
        )

    return dates, "csv_calendar"


# ============================================================
# 8. 固定亂數種子
# ============================================================

def seed_everything(seed):
    random.seed(seed)

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(seed)


def experiment_seed(market, origin):
    """
    同一市場、同一預測起點：
    A/B/C/D 使用相同起始亂數種子。
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
    )


# ============================================================
# 9. Kronos AI 預測
# ============================================================


def predict(
    predictor,
    history,
    future_dates,
    sample_count,
    seed
):
    """
    Kronos V3.1 prediction.

    Historical OHLC: strict validation.
    Predicted OHLC: preserve raw values and flag violations.
    Closing-price backtest may continue if close is valid.
    """

    # 1. Check historical input
    validate_ohlc(
        history,
        "Historical input"
    )

    if len(future_dates) != HORIZON:
        raise ValueError(
            f"Expected {HORIZON} future dates"
        )

    seed_everything(seed)

    x_dates = pd.Series(
        pd.DatetimeIndex(history.index)
    ).reset_index(drop=True)

    y_dates = pd.Series(
        pd.DatetimeIndex(future_dates)
    ).reset_index(drop=True)

    # 2. Run Kronos
    with torch.inference_mode():
        result = predictor.predict(
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
        not isinstance(result, pd.DataFrame)
        or len(result) != HORIZON
    ):
        raise RuntimeError(
            "Kronos returned unexpected "
            "forecast shape"
        )

    if not set(OHLC).issubset(result.columns):
        raise RuntimeError(
            "Kronos forecast missing OHLC columns"
        )

    result = (
        result[OHLC]
        .copy()
        .reset_index(drop=True)
    )

    for column in OHLC:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # 3. Closing price must be finite and positive
    close_array = result[
        "close"
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(close_array).all()
        or (close_array <= 0).any()
    ):
        raise RuntimeError(
            "Kronos predicted close is invalid; "
            "cannot calculate return"
        )

    # 4. Diagnose raw predicted OHLC
    price_array = result[
        OHLC
    ].to_numpy(dtype=float)

    finite_positive = (
        np.isfinite(price_array).all(axis=1)
        & (price_array > 0).all(axis=1)
    )

    # Allow tiny floating-point rounding differences
    tolerance = (
        result["close"].abs() * 0.000001
    ).clip(lower=0.01)

    expected_high_min = result[
        ["open", "low", "close"]
    ].max(axis=1)

    expected_low_max = result[
        ["open", "high", "close"]
    ].min(axis=1)

    high_valid = (
        result["high"] + tolerance
        >= expected_high_min
    )

    low_valid = (
        result["low"] - tolerance
        <= expected_low_max
    )

    ohlc_valid = (
        finite_positive
        & high_valid.to_numpy(dtype=bool)
        & low_valid.to_numpy(dtype=bool)
    )

    result["ohlc_valid"] = ohlc_valid

    # 5. Record invalid candles without modifying prices
    invalid_count = int(
        (~result["ohlc_valid"]).sum()
    )

    if invalid_count > 0:
        log(
            "[PREDICTION QUALITY WARNING] "
            f"origin={history.index[-1].date()}, "
            f"samples={sample_count}, "
            f"seed={seed}, "
            f"invalid_ohlc={invalid_count}/{HORIZON}"
        )

        examples = result.loc[
            ~result["ohlc_valid"],
            OHLC
        ].head(3)

        log(
            "[INVALID OHLC EXAMPLES] "
            + str(
                examples.round(2).to_dict("records")
            )
        )

    # 6. Assign future dates
    result.index = pd.DatetimeIndex(
        future_dates
    )

    result.index.name = "date"

    # 7. Calculate forecast close-to-close returns
    reference = float(
        history["close"].iloc[-1]
    )

    result["return_pct"] = (
        100.0
        * (
            result["close"] / reference
            - 1.0
        )
    )

    return result



# ============================================================
# 10. 歷史異常門檻
# ============================================================

def historical_reference(taiex, tx):
    close = pd.concat(
        {
            "TAIEX": taiex.close,
            "TX": tx.close
        },
        axis=1
    ).dropna()

    rows = []

    for h in (1, 3, 5):
        r = close.pct_change(
            h,
            fill_method=None
        ).dropna() * 100

        rows.append({
            "day": h,
            "historical_abs_gap_p99_pp": float(
                (r.TX - r.TAIEX).abs().quantile(.99)
            ),
            "historical_abs_tx_return_p99_pct": float(
                r.TX.abs().quantile(.99)
            )
        })

    ref = pd.DataFrame(rows)

    ref.to_csv(
        OUTPUT / "historical_reference.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return ref


# ============================================================
# 11. 每日四組比較圖
# ============================================================

def plot_comparison(summary, market):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for scenario, df in summary.groupby(
        "scenario"
    ):
        ax.plot(
            df.day,
            df[f"{market}_return_pct"],
            "-o",
            label=scenario
        )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Return %")

    ax.set_title(
        f"{market} Kronos-base scenario comparison"
    )

    ax.legend()
    ax.grid(alpha=.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{market}_scenario_comparison.png",
        dpi=140
    )

    plt.close(fig)


# ============================================================
# 12. 每日四組預測
# ============================================================

def run_experiments(
    predictor,
    taiex,
    tx,
    future_dates,
    calendar_mode
):
    reference = historical_reference(
        taiex,
        tx
    ).set_index("day")

    results = []
    rows = []

    for label, count, samples in SCENARIOS:
        forecasts = {}

        for market, data in (
            ("TAIEX", taiex),
            ("TX", tx)
        ):
            hist = data.tail(count)

            if len(hist) != count:
                raise RuntimeError(
                    f"{market} requires {count} bars"
                )

            seed = experiment_seed(
                market,
                data.index[-1]
            )

            log(
                f"[LIVE] {label} {market} "
                f"context={count} "
                f"samples={samples} seed={seed}"
            )

            pred = predict(
                predictor,
                hist,
                future_dates,
                samples,
                seed
            )

            pred.to_csv(
                OUTPUT / f"{market}_{label}_forecast.csv",
                encoding="utf-8-sig"
            )

            if label == "D":
                pred.to_csv(
                    OUTPUT / f"{market}_forecast.csv",
                    encoding="utf-8-sig"
                )

            forecasts[market] = pred

            results.append({
                "scenario": label,
                "market": market,
                "lookback": count,
                "samples": samples,
                "seed": seed,
                "day_1_return_pct": float(
                    pred.return_pct.iloc[0]
                ),
                "day_3_return_pct": float(
                    pred.return_pct.iloc[2]
                ),
                "day_5_return_pct": float(
                    pred.return_pct.iloc[4]
                )
            })

        for h in (1, 3, 5):
            a = float(
                forecasts["TAIEX"].return_pct.iloc[h-1]
            )

            b = float(
                forecasts["TX"].return_pct.iloc[h-1]
            )

            gap = b - a
            ref = reference.loc[h]

            warn = (
                abs(gap)
                > ref.historical_abs_gap_p99_pp
                or abs(b)
                > ref.historical_abs_tx_return_p99_pct
            )

            rows.append({
                "scenario": label,
                "lookback": count,
                "sample_count": samples,
                "day": h,
                "forecast_date": str(
                    pd.Timestamp(
                        future_dates[h-1]
                    ).date()
                ),
                "TAIEX_return_pct": a,
                "TX_return_pct": b,
                "return_gap_pp": gap,
                "review_status": (
                    "REVIEW" if warn else "OK"
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
        "# Kronos-base V3 每日參數敏感度比較",
        "",
        (
            f"資料截止日：{taiex.index[-1].date()}"
            f"；日期模式：{calendar_mode}"
        ),
        "",
        "不是歷史回測；REVIEW 不是準確率指標。",
        "",
        "| 情境 | 日數 | TAIEX % | TX % | 差距 pp | 檢查 |",
        "|---|---:|---:|---:|---:|---|"
    ]

    for r in summary.itertuples(index=False):
        report.append(
            f"| {r.scenario} | {r.day} | "
            f"{r.TAIEX_return_pct:+.2f} | "
            f"{r.TX_return_pct:+.2f} | "
            f"{r.return_gap_pp:+.2f} | "
            f"{r.review_status} |"
        )

    report += [
        "",
        "## 參數敏感度",
        (
            "固定種子不代表一定能精確重現"
            "不同硬體的結果。"
        )
    ]

    (
        OUTPUT / "experiment_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    return results, {
        "scenario_count": 4,
        "market_forecast_runs": 8,
        "comparisons": len(summary),
        "review_rows": int(
            (summary.review_status == "REVIEW").sum()
        )
    }


# ============================================================
# 13. TX 歷史實際報酬
# ============================================================

def realized_tx_return(
    origin,
    target,
    selected_all,
    combined
):
    """
    以目標日為截止點建立轉倉中性序列，
    用來衡量已實現報酬。

    未來資料僅用於評分，絕不進入預測輸入。
    """

    selected = selected_all[
        selected_all.date <= target
    ].tail(400)

    adjusted, _ = back_adjust_tx(
        selected,
        combined[
            combined.date <= target
        ]
    )

    points = adjusted.set_index(
        "date"
    ).close

    return float(
        100.0
        * (
            points.loc[target]
            / points.loc[origin]
            - 1.0
        )
    )


# ============================================================
# 14. 歷史回測統計
# ============================================================

def backtest_summary(detail):
    rows = []

    for (
        market,
        scenario,
        h
    ), g in detail.groupby([
        "market",
        "scenario",
        "horizon"
    ]):
        error = (
            g.pred_return_pct
            - g.actual_return_pct
        )

        flat_error = (
            g.actual_return_pct.abs()
        )

        momentum_error = (
            g.momentum_return_pct
            - g.actual_return_pct
        ).abs()

        direction_accuracy = float(
            (
                np.sign(g.pred_return_pct)
                == np.sign(g.actual_return_pct)
            ).mean()
        )

        momentum_direction_accuracy = float(
            (
                np.sign(g.momentum_return_pct)
                == np.sign(g.actual_return_pct)
            ).mean()
        )

        rows.append({
            "market": market,
            "scenario": scenario,
            "horizon": int(h),
            "lookback": int(g.lookback.iloc[0]),
            "samples": int(g.samples.iloc[0]),
            "n": len(g),
            "direction_accuracy": direction_accuracy,
            "mae_pp": float(error.abs().mean()),
            "rmse_pp": float(
                np.sqrt((error**2).mean())
            ),
            "flat_mae_pp": float(
                flat_error.mean()
            ),
            "momentum_mae_pp": float(
                momentum_error.mean()
            ),
            "momentum_direction_accuracy":
                momentum_direction_accuracy,
            "beats_flat_mae": bool(
                error.abs().mean()
                < flat_error.mean()
            )
        })

    return pd.DataFrame(rows).sort_values([
        "market",
        "horizon",
        "scenario"
    ])


# ============================================================
# 15. 歷史滾動回測核心
# ============================================================

def run_backtest(
    predictor,
    taiex,
    combined,
    days
):
    """
    每個預測起點重新建立 TX 連續合約。

    不使用預測日以後的 TX 價格或轉倉資訊
    來形成模型輸入。
    """

    selected_all = select_front_contract(
        combined
    )

    common = taiex.index.intersection(
        pd.DatetimeIndex(
            selected_all.date
        )
    ).sort_values()

    if len(common) < 410:
        raise RuntimeError(
            "Not enough aligned history "
            f"for backtest: {len(common)}"
        )

    origins = common[
        399:-HORIZON
    ][-days:]

    if len(origins) < days:
        raise RuntimeError(
            f"Only {len(origins)} available origins; "
            f"requested {days}"
        )

    rows = []

    log(
        f"[BACKTEST] {len(origins)} origins, "
        f"{len(SCENARIOS)*2*len(origins)} model calls"
    )

    for i, origin in enumerate(origins, 1):
        t0 = time.monotonic()

        position = common.get_loc(origin)

        future = common[
            position + 1:
            position + 1 + HORIZON
        ]

        # 歷史當下已知的 TX 資料
        tx_hist, _, _, _ = tx_asof(
            combined,
            selected_all,
            origin,
            400
        )

        histories = {
            "TAIEX": taiex.loc[:origin].tail(400),
            "TX": tx_hist
        }

        actual = {}

        for market in ("TAIEX", "TX"):
            actual[market] = {}

            for h in (1, 3, 5):
                target = future[h-1]

                if market == "TAIEX":
                    value = (
                        100.0
                        * (
                            float(
                                taiex.at[target, "close"]
                            )
                            / float(
                                taiex.at[origin, "close"]
                            )
                            - 1.0
                        )
                    )
                else:
                    value = realized_tx_return(
                        origin,
                        target,
                        selected_all,
                        combined
                    )

                actual[market][h] = float(value)

        for label, lookback, samples in SCENARIOS:
            for market in ("TAIEX", "TX"):
                hist = histories[
                    market
                ].tail(lookback)

                if len(hist) < lookback:
                    raise RuntimeError(
                        "Insufficient "
                        f"{market} history at {origin}"
                    )

                seed = experiment_seed(
                    market,
                    origin
                )

                p = predict(
                    predictor,
                    hist,
                    future,
                    samples,
                    seed
                )

                momentum5 = (
                    100.0
                    * (
                        float(hist.close.iloc[-1])
                        / float(hist.close.iloc[-6])
                        - 1.0
                    )
                )

                for h in (1, 3, 5):
                    rows.append({
                        "origin": str(
                            origin.date()
                        ),
                        "target_date": str(
                            future[h-1].date()
                        ),
                        "market": market,
                        "scenario": label,
                        "lookback": lookback,
                        "samples": samples,
                        "horizon": h,
                        "random_seed": seed,
                        "pred_return_pct": float(
                            p.return_pct.iloc[h-1]
                        ),
                        "actual_return_pct":
                            actual[market][h],
                        "momentum_return_pct":
                            momentum5*h/5.0
                    })

        # 每 5 個日期保存一次中間結果
        if i % 5 == 0 or i == len(origins):
            pd.DataFrame(rows).to_csv(
                OUTPUT / "backtest_detail.csv",
                index=False,
                encoding="utf-8-sig"
            )

        log(
            f"[BACKTEST] {i}/{len(origins)} "
            f"origin={origin.date()} "
            f"took {time.monotonic()-t0:.1f}s"
        )

    detail = pd.DataFrame(rows)

    summary = backtest_summary(detail)

    summary.to_csv(
        OUTPUT / "backtest_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 回測報告
    # ========================================================

    report = [
        "# Kronos-base V3 滾動日期歷史回測",
        "",
        (
            f"共 {len(origins)} 個歷史預測起點；"
            f"模型推論 {len(origins)*8} 次。"
        ),
        "",
        (
            "重要：不保證不在 Kronos 預訓練資料"
            "範圍內，因此不能宣稱完全樣本外。"
        ),
        (
            "TX 報酬使用截止目標日的轉倉中性"
            "序列評分；非實際可成交報酬。"
        ),
        "",
        (
            "| 市場 | 情境 | 期數 | 樣本 | "
            "方向正確率 | MAE(pp) | "
            "Flat MAE(pp) | 動能 MAE(pp) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]

    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.market} | "
            f"{row.scenario} | "
            f"{row.horizon} | "
            f"{row.n} | "
            f"{row.direction_accuracy:.1%} | "
            f"{row.mae_pp:.3f} | "
            f"{row.flat_mae_pp:.3f} | "
            f"{row.momentum_mae_pp:.3f} |"
        )

    (
        OUTPUT / "backtest_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    # ========================================================
    # TX 歷史回測 MAE 圖表
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    tx_summary = summary[
        summary.market == "TX"
    ]

    for scenario, group in tx_summary.groupby(
        "scenario"
    ):
        ax.plot(
            group.horizon,
            group.mae_pp,
            marker="o",
            label=f"TX {scenario}"
        )

    ax.set_title(
        "TX Kronos-base V3 rolling-backtest MAE"
    )

    ax.set_xticks([1, 3, 5])
    ax.set_ylabel(
        "MAE (percentage points)"
    )
    ax.set_xlabel(
        "Forecast horizon (trading days)"
    )

    ax.grid(alpha=.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "backtest_tx_mae.png",
        dpi=145
    )

    plt.close(fig)

    return {
        "origins": len(origins),
        "prediction_calls": 8*len(origins),
        "detail_rows": len(detail),
        "summary_rows": len(summary),
        "dates": [
            str(origins[0].date()),
            str(origins[-1].date())
        ]
    }


# ============================================================
# 16. 主程式
# ============================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    start = time.monotonic()

    status = {
        "success": False,
        "version": "V3",
        "mode": MODE,
        "model": MODEL_NAME,
        "created_at_taipei":
            datetime.now(TZ).isoformat(),
        "config": {
            "backtest_days": BACKTEST_DAYS,
            "seed": BASE_SEED,
            "scenarios": SCENARIOS
        }
    }

    try:
        if (
            MODE not in ("live", "backtest")
            or not (1 <= BACKTEST_DAYS <= 100)
        ):
            raise ValueError(
                "KRONOS_MODE=live|backtest; "
                "BACKTEST_DAYS=1..100"
            )

        log(
            f"[V3] mode={MODE}; "
            f"model={MODEL_NAME}"
        )

        # ----------------------------------------------------
        # 下載兩市場資料
        # ----------------------------------------------------

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
                selected.date.iloc[-1],
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

            dates, calendar_mode = future_market_dates(
                taiex.index[-1]
            )

        # ----------------------------------------------------
        # 載入 Kronos-base
        # ----------------------------------------------------

        log(
            f"[MODEL] Loading {MODEL_NAME} on CPU"
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

        # ----------------------------------------------------
        # 每日預測模式
        # ----------------------------------------------------

        if MODE == "live":
            results, diag = run_experiments(
                predictor,
                taiex.tail(400),
                tx,
                dates,
                calendar_mode
            )

            status.update({
                "results": results,
                "diagnostics": diag,
                "last_date": str(
                    taiex.index[-1].date()
                ),
                "night_coverage": coverage,
                "calendar_mode": calendar_mode
            })

        # ----------------------------------------------------
        # 歷史回測模式
        # ----------------------------------------------------

        else:
            status["backtest"] = run_backtest(
                predictor,
                taiex,
                combined,
                BACKTEST_DAYS
            )

        status["success"] = True

        log(
            "[V3 SUCCESS] Completed"
        )

    except Exception as exc:
        status["error_type"] = (
            type(exc).__name__
        )

        status["error"] = str(exc)

        status["traceback"] = traceback.format_exc(
            limit=10
        )

        log(
            f"[V3 FAILED] {exc}"
        )

        raise

    finally:
        status["elapsed_seconds"] = round(
            time.monotonic()-start,
            1
        )

        save_json(
            "status.json",
            status
        )


if __name__ == "__main__":
    main()
