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


def predict_market(predictor, name: str, frame: pd.DataFrame, future_dates) -> dict:
    log(f"[MODEL] predicting {name}")
    start = time.monotonic()
    with torch.inference_mode():
        pred = predictor.predict(
            df=frame[OHLC].reset_index(drop=True),
            x_timestamp=pd.Series(pd.DatetimeIndex(frame.index)).reset_index(drop=True),
            y_timestamp=pd.Series(pd.DatetimeIndex(future_dates)).reset_index(drop=True),
            pred_len=HORIZON,
            T=1.0, top_p=0.9,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
    if len(pred) != HORIZON or any(x not in pred.columns for x in OHLC):
        raise RuntimeError(f"Kronos returned invalid {name} forecast shape/columns")
    pred = pred[OHLC].copy().reset_index(drop=True)
    for col in OHLC:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    validate_ohlc(pred, f"{name} forecast")
    pred.index = pd.DatetimeIndex(future_dates)
    pred.index.name = "date"
    last_close = float(frame["close"].iloc[-1])
    pred["return_pct"] = (pred["close"] / last_close - 1.0) * 100
    pred.to_csv(OUTPUT / f"{name}_forecast.csv", encoding="utf-8-sig")
    history = frame["close"].tail(60)
    linked = pd.concat([
        pd.Series([last_close], index=[frame.index[-1]]), pred["close"]
    ])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history.index, history.to_numpy(), label="Historical close")
    ax.plot(linked.index, linked.to_numpy(), "--o", label="Kronos forecast")
    ax.set_title(f"Kronos-base {name}: daily forecast")
    ax.set_ylabel("Index points")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_forecast.png", dpi=150)
    plt.close(fig)
    summary = {
        "market": name,
        "last_date": str(frame.index[-1].date()),
        "last_close": last_close,
        "forecast": {
            f"day_{d}": {
                "date": str(pred.index[d-1].date()),
                "close": float(pred["close"].iloc[d-1]),
                "return_pct": float(pred["return_pct"].iloc[d-1]),
            }
            for d in (1, 3, 5) if d <= HORIZON
        },
        "prediction_seconds": round(time.monotonic() - start, 2),
    }
    log(f"[MODEL] finished {name}: {summary['forecast']}")
    return summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {
        "success": False,
        "created_at_taipei": datetime.now(TZ).isoformat(),
        "model": MODEL_NAME,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "sample_count": SAMPLE_COUNT,
        "data_source": {"TAIEX": "Yahoo ^TWII", "TX": "FinMind TaiwanFuturesDaily"},
    }
    try:
        if not (1 <= LOOKBACK <= 512):
            raise ValueError("LOOKBACK must be between 1 and 512 for Kronos-base")
        if HORIZON < 5:
            raise ValueError("HORIZON must be >= 5 for day 1/3/5 report")
        if SAMPLE_COUNT < 1:
            raise ValueError("SAMPLE_COUNT must be >= 1")
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
        device = "cpu"
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        tokenizer.eval()
        model.eval()
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

        log("[5/5] Forecasting both markets")
        status["results"] = [
            predict_market(predictor, "TAIEX", taiex, dates),
            predict_market(predictor, "TX", tx, dates),
        ]
        status["success"] = True
        log("[SUCCESS] Both market forecasts complete")
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
