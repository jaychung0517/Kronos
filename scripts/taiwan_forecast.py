"""Kronos-base v2: TAIEX and TX day/night four-scenario forecast experiment.

Run from the official Kronos repository root:
    python -m scripts.taiwan_forecast

Dependencies beyond upstream Kronos requirements: yfinance, requests, matplotlib.
FinMind dataset: TaiwanFuturesDaily, data_id=TX.

This program is for research; continuous futures use backward additive adjustments.
Future dates fall back to weekdays if data/trading_calendar.csv is not provided.
This is a parameter sensitivity comparison, NOT a historical accuracy backtest.
"""

from __future__ import annotations

import json
import random
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
LOOKBACK = 400  # The largest required historical window (also required for TX assembly).
HORIZON = 5
EXPERIMENTS = [
    ("A", 200, 1),
    ("B", 200, 5),
    ("C", 400, 1),
    ("D", 400, 5),
]
BASE_SEED = int(os.getenv("KRONOS_SEED", "20261104"))
STRICT_CALENDAR = os.getenv("STRICT_TRADING_CALENDAR", "0").lower() in ("true", "1", "yes")
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
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != 200:
                raise RuntimeError(
                    f"FinMind error status={payload.get('status')}, "
                    f"msg={str(payload.get('msg', ''))[:240]}"
                )
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise RuntimeError("FinMind response data is not a list")
            return pd.DataFrame(rows)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"FinMind request failed: {last_error}")


def fetch_finmind() -> pd.DataFrame:
    """Retrieve in ~180-day chunks to avoid massive responses and aid retries."""
    end = datetime.now(TZ).date()
    start = end - timedelta(days=950)
    pieces = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=179))
        log(f"[TX] FinMind {current} to {chunk_end}")
        part = request_finmind(str(current), str(chunk_end))
        if not part.empty:
            pieces.append(part)
        current = chunk_end + timedelta(days=1)
    if not pieces:
        raise RuntimeError("FinMind returned no TX records")
    result = pd.concat(pieces, ignore_index=True)
    log(f"[TX] downloaded {len(result)} raw rows")
    return result


def fetch_taiex() -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            frame = yf.download(
                "^TWII", period="5y", interval="1d",
                auto_adjust=False, progress=False,
                multi_level_index=False, threads=False,
            )
            if frame is None or frame.empty:
                raise RuntimeError("Yahoo returned empty ^TWII data")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close"
            })
            if any(name not in frame for name in OHLC):
                raise RuntimeError(f"Unexpected Yahoo columns: {list(frame.columns)}")
            frame = frame[OHLC].copy()
            for col in OHLC:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.index = pd.to_datetime(frame.index)
            if frame.index.tz is not None:
                frame.index = frame.index.tz_convert(TZ).tz_localize(None)
            frame.index = frame.index.normalize()
            frame = frame.dropna().sort_index()
            if frame.index.has_duplicates:
                raise RuntimeError("TAIEX contains duplicate dates")
            if len(frame) < LOOKBACK:
                raise RuntimeError(f"TAIEX only has {len(frame)} bars; need {LOOKBACK}")
            frame = frame.tail(LOOKBACK)
            validate_ohlc(frame, "TAIEX")
            log(f"[TAIEX] {len(frame)} bars, last={frame.index[-1].date()}")
            return frame
        except Exception as exc:
            last_err = exc
            if attempt == 2:
                break
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"TAIEX download failed: {last_err}")


def make_tx_contract_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct one row per (trading date, delivery month) from official sessions.

    FinMind: after_market already belongs to trading date D; do NOT shift its date.
    For D, after_market is previous business day 15:00 through D 05:00,
    followed by D position 08:45-13:45.
    """
    required = {
        "date", "futures_id", "contract_date", "trading_session",
        "open", "max", "min", "close", "volume",
    }
    absent = required - set(raw.columns)
    if absent:
        raise RuntimeError(f"FinMind TX missing columns: {sorted(absent)}")

    df = raw.copy()
    df = df[df["futures_id"].astype(str).str.upper() == "TX"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["contract_date"] = df["contract_date"].astype(str).str.strip()
    df = df[df["contract_date"].str.fullmatch(r"20\d{4}")].copy()
    df = df[df["contract_date"].str[-2:].astype(int).between(1, 12)]
    df = df[df["trading_session"].isin(["position", "after_market"])].copy()
    for field in ("open", "max", "min", "close", "volume"):
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=["date", "open", "max", "min", "close", "volume"])
    df = df[
        (df[["open", "max", "min", "close"]] > 0).all(axis=1)
        & (df["volume"] > 0)
        & (df["max"] >= df[["open", "min", "close"]].max(axis=1))
        & (df["min"] <= df[["open", "max", "close"]].min(axis=1))
    ].copy()
    if df.empty:
        raise RuntimeError("No valid TX contract sessions after filtering")

    keys = ["date", "contract_date", "trading_session"]
    if df.duplicated(keys).any():
        duplicate_examples = df[df.duplicated(keys, keep=False)][keys].head(4)
        raise RuntimeError(f"Duplicate TX sessions: {duplicate_examples.to_dict('records')}")

    records = []
    for (date, contract), group in df.groupby(["date", "contract_date"], sort=True):
        day_rows = group[group["trading_session"] == "position"]
        night_rows = group[group["trading_session"] == "after_market"]
        if day_rows.empty:
            # A date is not considered completed if its day session does not exist.
            continue
        day = day_rows.iloc[0]
        has_night = not night_rows.empty
        night = night_rows.iloc[0] if has_night else None
        prices = [day] if night is None else [night, day]
        records.append({
            "date": pd.Timestamp(date),
            "contract_date": str(contract),
            "open": float(night["open"] if has_night else day["open"]),
            "high": float(max(x["max"] for x in prices)),
            "low": float(min(x["min"] for x in prices)),
            "close": float(day["close"]),
            "volume": float(sum(x["volume"] for x in prices)),
            "has_night": bool(has_night),
        })

    combined = pd.DataFrame(records)
    if combined.empty:
        raise RuntimeError("No TX contracts with completed daytime session")
    combined = combined.sort_values(["date", "contract_date"]).reset_index(drop=True)
    if combined.duplicated(["date", "contract_date"]).any():
        raise RuntimeError("Duplicate TX combined contract dates")
    validate_ohlc(combined, "TX combined contract bars")
    return combined


def select_front_contract(combined: pd.DataFrame) -> pd.DataFrame:
    """Select earliest available non-past delivery month per trading date."""
    available = combined.copy()
    available = available[
        available["contract_date"] >= available["date"].dt.strftime("%Y%m")
    ]
    selected = (
        available.sort_values(["date", "contract_date"])
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    if len(selected) < LOOKBACK:
        raise RuntimeError(f"TX has only {len(selected)} daily bars, need {LOOKBACK}")
    return selected.tail(LOOKBACK).reset_index(drop=True).copy()


def back_adjust_tx(selected: pd.DataFrame, combined: pd.DataFrame):
    """Additive back adjustment at each roll using same-date overlapping closes.

    Reference day: latest date on/before roll date when old AND new contracts
    both have completed day-session closes, at most MAX_OVERLAP_AGE_DAYS old.
    Each gap is added to all bars before the corresponding roll so that the
    most recent contract remains at its true (unadjusted) price level.
    """
    adjusted = selected.copy()
    quotes = combined.pivot(index="date", columns="contract_date", values="close").sort_index()
    log_rows = []
    for pos in range(1, len(adjusted)):
        old_c = str(adjusted.at[pos - 1, "contract_date"])
        new_c = str(adjusted.at[pos, "contract_date"])
        if old_c == new_c:
            continue
        date = pd.Timestamp(adjusted.at[pos, "date"])
        if new_c < old_c:
            raise RuntimeError(f"TX backwards roll on {date.date()}: {old_c}->{new_c}")
        if old_c not in quotes or new_c not in quotes:
            raise RuntimeError(f"TX roll reference quotes unavailable: {old_c}->{new_c}")
        both = quotes.loc[:date, [old_c, new_c]].dropna()
        if both.empty:
            raise RuntimeError(f"No overlapping TX quotes for {old_c}->{new_c} at {date.date()}")
        reference_date = both.index[-1]
        age = (date - reference_date).days
        if age > MAX_OVERLAP_AGE_DAYS:
            raise RuntimeError(
                f"TX rollover overlap too old: {old_c}->{new_c}, "
                f"roll={date.date()}, reference={reference_date.date()}"
            )
        gap = float(both.iloc[-1][new_c] - both.iloc[-1][old_c])
        # Sequential addition accumulates all later roll gaps into older bars.
        adjusted.loc[:pos - 1, OHLC] = adjusted.loc[:pos - 1, OHLC].add(gap)
        log_rows.append({
            "roll_date": date.date().isoformat(),
            "reference_date": reference_date.date().isoformat(),
            "old_contract": old_c,
            "new_contract": new_c,
            "adjustment_points": gap,
            "overlap_age_days": int(age),
        })
        log(f"[TX ROLL] {date.date()} {old_c}->{new_c}: {gap:+.1f} pts, ref={reference_date.date()}")

    validate_ohlc(adjusted, "TX adjusted")
    roll_df = pd.DataFrame(log_rows, columns=[
        "roll_date", "reference_date", "old_contract", "new_contract",
        "adjustment_points", "overlap_age_days",
    ])
    return adjusted, roll_df


def prepare_tx(raw: pd.DataFrame):
    combined = make_tx_contract_bars(raw)
    selected = select_front_contract(combined)
    adjusted, roll_log = back_adjust_tx(selected, combined)
    coverage = float(adjusted["has_night"].mean())
    log(f"[TX] bars={len(adjusted)}, night_coverage={coverage:.1%}, rolls={len(roll_log)}")
    if coverage < MIN_NIGHT_COVERAGE:
        raise RuntimeError(
            f"TX night-session coverage {coverage:.1%} is below "
            f"required {MIN_NIGHT_COVERAGE:.1%}; verify missing sessions"
        )
    if len(roll_log) == 0:
        log("[TX] WARNING: no rollover detected inside lookback")
    adjusted.to_csv(OUTPUT / "tx_continuous_adjusted.csv", index=False, encoding="utf-8-sig")
    roll_log.to_csv(OUTPUT / "tx_roll_log.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT / "tx_contract_bars.csv", index=False, encoding="utf-8-sig")
    frame = adjusted.set_index("date")[OHLC].copy()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise RuntimeError("TX adjusted dates are not unique and increasing")
    return frame, {
        "latest_contract": str(adjusted.iloc[-1]["contract_date"]),
        "roll_count": int(len(roll_log)),
        "night_coverage": round(coverage, 5),
        "day_only_dates": [str(d.date()) for d in adjusted.loc[~adjusted["has_night"], "date"]],
        "adjustment_method": "backward_additive",
    }


def check_last_dates(taiex: pd.DataFrame, tx: pd.DataFrame) -> None:
    date_a = pd.Timestamp(taiex.index[-1]).date()
    date_b = pd.Timestamp(tx.index[-1]).date()
    if date_a != date_b:
        raise RuntimeError(f"Market data dates differ: TAIEX={date_a}, TX={date_b}")
    age = (datetime.now(TZ).date() - date_a).days
    if age < 0 or age > MAX_STALE_DAYS:
        raise RuntimeError(f"Market data stale/future: {date_a}, {age} calendar days ago")
    log(f"[DATA] aligned through {date_a}; age={age} day(s)")


def future_market_dates(last_date: pd.Timestamp, horizon: int):
    """Use supplied trading calendar if present, else weekdays with warning.

    Accepted data/trading_calendar.csv format: date,is_open (true/false).
    File must include all needed future trading dates. Calendar for TWSE is only
    a proxy for TAIFEX where occasional schedule differences may apply.
    """
    if not CALENDAR_FILE.exists():
        if STRICT_CALENDAR:
            raise RuntimeError("STRICT_TRADING_CALENDAR=1 but data/trading_calendar.csv is missing")
        log("[CALENDAR] WARNING: no calendar CSV; weekdays-only forecast dates (holidays may be wrong)")
        return pd.bdate_range(
            pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon
        ), "weekdays_only"
    cal = pd.read_csv(CALENDAR_FILE, dtype={"is_open": "string"})
    if not {"date", "is_open"}.issubset(cal.columns):
        raise RuntimeError("trading_calendar.csv must have date and is_open columns")
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    if cal["date"].isna().any() or cal["date"].duplicated().any():
        raise RuntimeError("trading_calendar.csv has invalid or duplicate dates")
    statuses = cal["is_open"].str.lower().str.strip()
    if not statuses.isin(["true", "false", "1", "0"]).all():
        raise RuntimeError("is_open must be true or false")
    cal["is_open"] = statuses.isin(["true", "1"])
    future = cal[(cal["date"] > pd.Timestamp(last_date)) & cal["is_open"]]
    future = future.sort_values("date").head(horizon)
    if len(future) < horizon:
        raise RuntimeError("trading_calendar.csv does not cover all requested future dates")
    log("[CALENDAR] using supplied trading_calendar.csv")
    return pd.DatetimeIndex(future["date"]), "csv_calendar"


def seed_everything(seed: int) -> None:
    """Set common PRNGs; numerical reproducibility still depends on libraries/CPU."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_single_market(frame: pd.DataFrame, pred: pd.DataFrame, name: str) -> None:
    history = frame["close"].tail(60)
    last_close = float(frame["close"].iloc[-1])
    linked = pd.concat([pd.Series([last_close], index=[frame.index[-1]]), pred["close"]])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history.index, history.to_numpy(), label="Historical close")
    ax.plot(linked.index, linked.to_numpy(), "--o", label="Kronos D (400 / 5)")
    ax.set_title(f"Kronos-base {name}: Scenario D forecast")
    ax.set_ylabel("Index points")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_forecast.png", dpi=150)
    plt.close(fig)


def predict_market(predictor, name: str, frame: pd.DataFrame,
                   future_dates, scenario: str, lookback: int, samples: int):
    """Run one predefined scenario; returns complete forecast and summary."""
    history = frame[OHLC].tail(lookback)
    if len(history) < lookback:
        raise RuntimeError(f"{name} has {len(history)} bars, scenario needs {lookback}")
    # Same seed for 1-vs-5 samples with identical input window, per market.
    seed = BASE_SEED + lookback * 10 + (0 if name == "TAIEX" else 1)
    seed_everything(seed)
    log(f"[EXPERIMENT {scenario}] {name}: lookback={lookback}, "
        f"sample_count={samples}, seed={seed}")
    started = time.monotonic()
    with torch.inference_mode():
        pred = predictor.predict(
            df=history.reset_index(drop=True),
            x_timestamp=pd.Series(pd.DatetimeIndex(history.index)).reset_index(drop=True),
            y_timestamp=pd.Series(pd.DatetimeIndex(future_dates)).reset_index(drop=True),
            pred_len=HORIZON,
            T=1.0, top_p=0.9,
            sample_count=samples,
            verbose=False,
        )
    if not isinstance(pred, pd.DataFrame) or len(pred) != HORIZON:
        raise RuntimeError(f"{name} scenario {scenario}: invalid predictor result length/type")
    if any(x not in pred.columns for x in OHLC):
        raise RuntimeError(f"{name} scenario {scenario}: prediction missing OHLC")
    pred = pred[OHLC].copy().reset_index(drop=True)
    for col in OHLC:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    validate_ohlc(pred, f"{name} scenario {scenario} forecast")
    pred.index = pd.DatetimeIndex(future_dates)
    pred.index.name = "date"
    last_close = float(frame["close"].iloc[-1])
    pred["return_pct"] = (pred["close"] / last_close - 1.0) * 100.0
    pred.to_csv(OUTPUT / f"{name}_{scenario}_forecast.csv", encoding="utf-8-sig")
    if scenario == "D":
        # Compatibility files: scenario D only, not a verified best choice.
        pred.to_csv(OUTPUT / f"{name}_forecast.csv", encoding="utf-8-sig")
        plot_single_market(frame, pred, name)
    summary = {
        "market": name,
        "scenario": scenario,
        "lookback": lookback,
        "sample_count": samples,
        "random_seed": seed,
        "last_date": str(frame.index[-1].date()),
        "last_close": last_close,
        "forecast": {
            f"day_{d}": {
                "date": str(pred.index[d - 1].date()),
                "close": float(pred["close"].iloc[d - 1]),
                "return_pct": float(pred["return_pct"].iloc[d - 1]),
            }
            for d in (1, 3, 5)
        },
        "prediction_seconds": round(time.monotonic() - started, 3),
    }
    return pred, summary


def historical_reference(taiex: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Descriptive 99th-percentile bounds, NOT calibrated forecast uncertainty."""
    aligned = pd.concat({
        "TAIEX": taiex["close"], "TX": tx["close"]
    }, axis=1, join="inner").dropna().sort_index()
    if len(aligned) < 150:
        raise RuntimeError(f"Only {len(aligned)} overlapping historical trading dates")
    log(f"[REFERENCE] common historical dates={len(aligned)}")
    rows = []
    for horizon in (1, 3, 5):
        r = aligned.pct_change(periods=horizon, fill_method=None).dropna() * 100.0
        gap = (r["TX"] - r["TAIEX"]).abs()
        tx_abs = r["TX"].abs()
        rows.append({
            "day": horizon,
            "observations": len(r),
            "historical_abs_gap_p99_pp": float(gap.quantile(.99)),
            "historical_abs_tx_return_p99_pct": float(tx_abs.quantile(.99)),
            "historical_max_abs_gap_pp": float(gap.max()),
            "historical_max_abs_tx_return_pct": float(tx_abs.max()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "historical_reference.csv", index=False, encoding="utf-8-sig")
    return result


def plot_scenario_comparison(summary_df: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario, group in summary_df.groupby("scenario", sort=True):
        ax.plot(group["day"], group[f"{name}_return_pct"], marker="o", label=f"{scenario} ({int(group.iloc[0]['lookback'])} / {int(group.iloc[0]['sample_count'])})")
    ax.axhline(0, linewidth=0.7, color="gray")
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Cumulative predicted return (%)")
    ax.set_title(f"Kronos-base {name}: 4 scenario comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_scenario_comparison.png", dpi=160)
    plt.close(fig)


def create_report(summary_df: pd.DataFrame, ref_df: pd.DataFrame, calendar_mode: str) -> None:
    header = [
        "# Kronos-base 台灣市場四組對照實驗",
        "",
        "此報告屬於單一收盤日的**參數敏感度對照**，不是歷史回測或可交易訊號。",
        "",
        f"- 模型：`{MODEL_NAME}`；預測天數：{HORIZON}",
        f"- 亂數種子基準值：{BASE_SEED}；日期模式：`{calendar_mode}`",
        "- A=200根/1次抽樣；B=200根/5次；C=400根/1次；D=400根/5次",
        "- 官方 Kronos 的 sample_count=5 會**先對五條抽樣路徑取平均**，此輸出不包含個別路徑分布。",
        "- 報表 `REVIEW` 表示超過有限歷史樣本的第99百分位參考範圍；非預測錯誤證明，也非預測機率。",
        "- TX 為後調整連續合約，不可直接將歷史絕對價位視為可成交報價。",
        "",
        "## 四組結果（累積預測漲跌幅）",
        "",
        "| 組別 | 日數 | TAIEX % | TX % | 差距（百分點） | 檢查 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summary_df.itertuples(index=False):
        header.append(f"| {r.scenario} | {r.day} | {r.TAIEX_return_pct:+.2f} | {r.TX_return_pct:+.2f} | {r.return_gap_pp:+.2f} | {r.review_status} |")
    header.extend([
        "", "## 重要限制", "",
        "1. 無交易日曆時，預測日期暫用平日，未排除台灣休市日。",
        "2. 分歧與極端報酬門檻是歷史描述統計，不代表模型可信區間。",
        "3. 固定亂數種子便於比較，但不同硬體、PyTorch 或依賴版本仍可能有差異。",
        "4. 單日的四組結果不能評估哪個參數最準，必須累積實際結果並做時間序列樣本外回測。",
        "5. TX 夜盤缺漏與換月仍應核對期交所行情、結算日與交易日曆。",
        "",
        "## 歷史參考範圍", "",
        "| 期間 | 有效樣本數 | 期現貨報酬差絕對值 P99（百分點） | TX 絕對報酬 P99（%） |",
        "|---|---:|---:|---:|",
    ])
    for r in ref_df.itertuples(index=False):
        header.append(f"| {r.day} 日 | {r.observations} | {r.historical_abs_gap_p99_pp:.3f} | {r.historical_abs_tx_return_p99_pct:.3f} |")
    (OUTPUT / "experiment_report.md").write_text("\n".join(header)+"\n", encoding="utf-8")


def run_experiments(predictor, taiex: pd.DataFrame, tx: pd.DataFrame,
                    future_dates, calendar_mode: str):
    reference = historical_reference(taiex, tx)
    refs = reference.set_index("day")
    results = []
    summary_rows = []
    for scenario, lookback, samples in EXPERIMENTS:
        predictions = {}
        for name, frame in (("TAIEX", taiex), ("TX", tx)):
            pred, result = predict_market(predictor, name, frame, future_dates,
                                          scenario, lookback, samples)
            predictions[name] = pred
            results.append(result)
        for day in (1, 3, 5):
            a = float(predictions["TAIEX"]["return_pct"].iloc[day - 1])
            b = float(predictions["TX"]["return_pct"].iloc[day - 1])
            gap = b - a
            reference_row = refs.loc[day]
            gap_flag = bool(abs(gap) > reference_row["historical_abs_gap_p99_pp"])
            tail_flag = bool(abs(b) > reference_row["historical_abs_tx_return_p99_pct"])
            status = "REVIEW" if gap_flag or tail_flag else "OK"
            summary_rows.append({
                "scenario": scenario,
                "lookback": lookback,
                "sample_count": samples,
                "day": day,
                "forecast_date": str(pd.Timestamp(future_dates[day - 1]).date()),
                "TAIEX_last_close": float(taiex["close"].iloc[-1]),
                "TAIEX_pred_close": float(predictions["TAIEX"]["close"].iloc[day - 1]),
                "TAIEX_return_pct": a,
                "TX_last_close": float(tx["close"].iloc[-1]),
                "TX_pred_close_adjusted_basis": float(predictions["TX"]["close"].iloc[day - 1]),
                "TX_return_pct": b,
                "return_gap_pp": gap,
                "historical_abs_gap_p99_pp": float(reference_row["historical_abs_gap_p99_pp"]),
                "historical_abs_tx_return_p99_pct": float(reference_row["historical_abs_tx_return_p99_pct"]),
                "gap_flag": gap_flag,
                "tx_tail_flag": tail_flag,
                "review_status": status,
            })
            log(f"[COMPARE] {scenario} D{day}: TAIEX={a:+.2f}%, TX={b:+.2f}%, "
                f"gap={gap:+.2f}pp -> {status}")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    for name in ("TAIEX", "TX"):
        plot_scenario_comparison(summary_df, name)
    create_report(summary_df, reference, calendar_mode)
    return results, {
        "scenario_count": len(EXPERIMENTS),
        "market_forecast_runs": len(results),
        "comparisons": len(summary_df),
        "review_rows": int((summary_df["review_status"] == "REVIEW").sum()),
        "report_files": ["experiment_summary.csv", "experiment_report.md",
                         "historical_reference.csv", "TAIEX_scenario_comparison.png",
                         "TX_scenario_comparison.png"],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {
        "success": False,
        "created_at_taipei": datetime.now(TZ).isoformat(),
        "model": MODEL_NAME,
        "horizon": HORIZON,
        "experiment_config": [
            {"scenario": s, "lookback": n, "sample_count": m}
            for s, n, m in EXPERIMENTS
        ],
        "base_seed": BASE_SEED,
        "data_source": {"TAIEX": "Yahoo ^TWII", "TX": "FinMind TaiwanFuturesDaily"},
        "warning": "Four-scenario sensitivity comparison, NOT verified forecast accuracy",
        "software": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    try:
        if HORIZON != 5:
            raise ValueError("This version runs exactly 5 forecast trading days")
        if max(n for _, n, _ in EXPERIMENTS) > 512:
            raise ValueError("Kronos maximum context is 512 bars")
        if not (0 <= MIN_NIGHT_COVERAGE <= 1):
            raise ValueError("MIN_NIGHT_COVERAGE must be in [0,1]")

        log("[1/5] Downloading TAIEX")
        taiex = fetch_taiex()
        taiex.to_csv(OUTPUT / "taiex_history.csv", encoding="utf-8-sig")

        log("[2/5] Downloading TX from FinMind")
        raw_tx = fetch_finmind()
        log("[3/5] Combining TX day/night and backward adjusting rolls")
        tx, tx_metadata = prepare_tx(raw_tx)
        status["tx_metadata"] = tx_metadata
        check_last_dates(taiex, tx)
        dates, calendar_mode = future_market_dates(taiex.index[-1], HORIZON)
        status["forecast_calendar_mode"] = calendar_mode
        status["data_last_date"] = str(taiex.index[-1].date())

        log(f"[4/5] Loading {MODEL_NAME} on CPU")
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        tokenizer.eval()
        model.eval()
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

        log("[5/5] Running 4 scenarios x 2 markets")
        status["results"], status["experiment_diagnostics"] = run_experiments(
            predictor, taiex, tx, dates, calendar_mode
        )
        status["success"] = True
        log("[SUCCESS] Four-scenario comparison complete")
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc(limit=8)
        log(f"[FAILED] {type(exc).__name__}: {exc}")
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json("status.json", status)


if __name__ == "__main__":
    main()
