"""Kronos V3.5-A: isolated four-market historical availability audit.

Run: python -m scripts.kronos_v35_data_audit --start 2017-05-16
This research tool does not import production models or send trade/Telegram orders.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, time as walltime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SYMBOLS = {"TAIEX": "^TWII", "NASDAQ": "^IXIC", "SOX": "^SOX"}
OHLC = ["open", "high", "low", "close"]
TX_FIELDS = ["date", "futures_id", "contract_date", "trading_session",
             "open", "max", "min", "close", "volume"]
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TPE = ZoneInfo("Asia/Taipei")
NY = ZoneInfo("America/New_York")


def log(message):
    print("[V35 AUDIT] " + message, flush=True)


def valid_ohlc_rows(df):
    values = df[OHLC].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    upper = np.maximum.reduce([df.open.to_numpy(), df.low.to_numpy(), df.close.to_numpy()])
    lower = np.minimum.reduce([df.open.to_numpy(), df.high.to_numpy(), df.close.to_numpy()])
    return pd.Series(finite & (df.high.to_numpy() >= upper) &
                     (df.low.to_numpy() <= lower), index=df.index)


def normalize_yahoo(raw, market):
    if raw is None or raw.empty:
        raise ValueError(f"{market}: Yahoo returned empty OHLC")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    if not set(OHLC).issubset(df.columns):
        raise ValueError(f"{market}: missing OHLC: {list(df.columns)}")
    df = df[OHLC].apply(pd.to_numeric, errors="coerce").dropna()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Taipei" if market == "TAIEX" else "America/New_York").tz_localize(None)
    df.index = idx.normalize()
    df = df.sort_index()
    if df.index.has_duplicates or not valid_ohlc_rows(df).all():
        raise ValueError(f"{market}: duplicated dates or invalid OHLC")
    if df.empty:
        raise ValueError(f"{market}: cleaned OHLC empty")
    df.index.name = "date"
    return df


def download_yahoo(market, symbol, start, end):
    import yfinance as yf
    last = None
    for attempt in range(3):
        try:
            raw = yf.download(symbol, start=str(start), end=str(end + timedelta(days=1)),
                              interval="1d", auto_adjust=False, progress=False,
                              multi_level_index=False, threads=False, timeout=45)
            df = normalize_yahoo(raw, market)
            log(f"{market}: {len(df)} daily bars, {df.index.min().date()}..{df.index.max().date()}")
            return df
        except Exception as exc:
            last = exc
            log(f"{market}: attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:150]}")
            if attempt < 2:
                time.sleep(3 * 2 ** attempt)
    raise RuntimeError(f"{market}: cannot download Yahoo: {last}")


def finmind_chunk(start, end, token, http=requests):
    headers = {"Authorization": "Bearer " + token} if token else {}
    params = {"dataset": "TaiwanFuturesDaily", "data_id": "TX",
              "start_date": str(start), "end_date": str(end)}
    response = http.get(FINMIND_URL, params=params, headers=headers, timeout=80)
    if response.status_code in (401, 402, 403, 429):
        raise RuntimeError(f"FinMind HTTP {response.status_code} (auth/quota)")
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 200 or not isinstance(payload.get("data"), list):
        raise RuntimeError(f"FinMind status={payload.get('status')}, msg={str(payload.get('msg'))[:120]}")
    return pd.DataFrame(payload["data"])


def download_tx(start, end, token, chunk_days=179, http=requests):
    if not 1 <= chunk_days <= 180:
        raise ValueError("chunk_days outside 1..180")
    fragments, failures = [], []
    current = start
    while current <= end:
        until = min(current + timedelta(days=chunk_days - 1), end)
        result, error = None, None
        for attempt in range(3):
            try:
                result = finmind_chunk(current, until, token, http)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:180]}"
                if any(code in error for code in ("HTTP 401", "HTTP 402", "HTTP 403", "HTTP 429")):
                    break
                if attempt < 2:
                    time.sleep(3 * 2 ** attempt)
        if result is not None:
            fragments.append(result)
            log(f"TX {current}..{until}: {len(result)} rows")
        else:
            failures.append({"source": "FinMind", "start": str(current),
                             "end": str(until), "error": error})
            log(f"TX {current}..{until}: FAILED {error}")
            if error and any(code in error for code in ("HTTP 401", "HTTP 402", "HTTP 403", "HTTP 429")):
                log("Stop requesting TX due to rate limit/auth; failure retained in report")
                break
        current = until + timedelta(days=1)
        time.sleep(0.12)
    return (pd.concat(fragments, ignore_index=True) if fragments
            else pd.DataFrame(columns=TX_FIELDS)), failures


def validate_tx(raw):
    if raw.empty:
        return pd.DataFrame(columns=TX_FIELDS), {"raw_rows": 0, "tx_rows": 0,
                  "invalid_rows": 0, "duplicate_rows": 0, "valid_session_rows": 0}
    missing = set(TX_FIELDS) - set(raw.columns)
    if missing:
        raise ValueError("TX missing schema: " + str(sorted(missing)))
    df = raw.loc[raw.futures_id.astype(str).str.upper().eq("TX"), TX_FIELDS].copy()
    count = len(df)
    df["date"] = pd.to_datetime(df.date, errors="coerce").dt.normalize()
    df["contract_date"] = df.contract_date.astype(str).str.strip()
    contract_ok = df.contract_date.str.fullmatch(r"20\d{4}").fillna(False)
    month_ok = pd.to_numeric(df.contract_date.str[-2:], errors="coerce").between(1, 12)
    session_ok = df.trading_session.isin(["position", "after_market"])
    for col in ("open", "max", "min", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ohlc_ok = valid_ohlc_rows(df.rename(columns={"max": "high", "min": "low"}))
    ok = (df.date.notna() & contract_ok & month_ok & session_ok & ohlc_ok &
          (df.volume > 0) & np.isfinite(df.volume))
    invalid = int((~ok).sum())
    df = df.loc[ok].copy()
    keys = ["date", "contract_date", "trading_session"]
    duplicate = df.duplicated(keys, keep=False)
    dup_count = int(duplicate.sum())
    # Ambiguous duplicates are excluded and counted, never silently repaired.
    df = df.loc[~duplicate].sort_values(keys).reset_index(drop=True)
    return df, {"raw_rows": len(raw), "tx_rows": count, "invalid_rows": invalid,
                "duplicate_rows": dup_count, "valid_session_rows": len(df)}


def tx_front_sessions(valid, start, end):
    columns = ["date", "front_contract", "day", "night", "complete"]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    df = valid[(valid.date.dt.date >= start) & (valid.date.dt.date <= end)].copy()
    df = df[df.contract_date >= df.date.dt.strftime("%Y%m")]
    rows = []
    for trading_day, group in df.groupby("date", sort=True):
        front = group.contract_date.min()
        selected = group[group.contract_date == front]
        day = bool(selected.trading_session.eq("position").any())
        night = bool(selected.trading_session.eq("after_market").any())
        rows.append({"date": trading_day, "front_contract": front, "day": day,
                     "night": night, "complete": day and night})
    return pd.DataFrame(rows, columns=columns)


def cross_market_asof(taiex_dates, nasdaq_dates, sox_dates):
    """Time zone/DST-aware no-lookahead alignment at Taiwan decision timestamps."""
    closes = {}
    for market, days in (("NASDAQ", nasdaq_dates), ("SOX", sox_dates)):
        closes[market] = sorted(
            (datetime.combine(pd.Timestamp(d).date(), walltime(16, 0), tzinfo=NY).astimezone(TPE),
             str(pd.Timestamp(d).date())) for d in days
        )
    rows = []
    for d in taiex_dates:
        trading_day = pd.Timestamp(d).date()
        for session, clock in (("TW_CLOSE", walltime(17, 10)), ("US_CLOSE", walltime(6, 20))):
            cutoff = datetime.combine(trading_day, clock, tzinfo=TPE)
            row = {"taiwan_date": str(trading_day), "decision": session,
                   "decision_time_tpe": cutoff.isoformat()}
            for market, sessions in closes.items():
                row[market + "_last_available"] = next(
                    (day for timestamp, day in reversed(sessions) if timestamp <= cutoff), "")
            rows.append(row)
    return pd.DataFrame(rows)


def year_counts(frames, front, start, end):
    rows = []
    for year in range(start.year, end.year + 1):
        row = {"year": year}
        for market, data in frames.items():
            row[market + "_days"] = int((data.index.year == year).sum()) if data is not None else 0
        year_data = front[front.date.dt.year == year] if not front.empty else pd.DataFrame()
        row["TX_front_days"] = len(year_data)
        row["TX_day_sessions"] = int(year_data.day.sum()) if len(year_data) else 0
        row["TX_night_sessions"] = int(year_data.night.sum()) if len(year_data) else 0
        row["TX_full_days"] = int(year_data.complete.sum()) if len(year_data) else 0
        row["TX_full_rate_on_front_days"] = round(float(year_data.complete.mean()), 4) if len(year_data) else None
        rows.append(row)
    return pd.DataFrame(rows)


def run(start, end, out, chunk_days=179):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "market_cache"
    cache.mkdir(exist_ok=True)
    begun = time.monotonic()
    frames, errors = {}, []
    for market, symbol in SYMBOLS.items():
        try:
            frames[market] = download_yahoo(market, symbol, start, end)
            frames[market].to_csv(cache / f"{market.lower()}_ohlc.csv.gz", compression="gzip")
        except Exception as exc:
            frames[market] = None
            errors.append({"source": market, "start": str(start), "end": str(end),
                           "error": str(exc)[:250]})
    tx, failures = download_tx(start, end, os.getenv("FINMIND_TOKEN", "").strip(),
                               chunk_days=chunk_days)
    errors.extend(failures)
    valid, quality = validate_tx(tx)
    if not valid.empty:
        valid.to_csv(cache / "tx_valid_sessions.csv.gz", index=False, compression="gzip")
    front = tx_front_sessions(valid, start, end)
    front.to_csv(out / "tx_front_sessions.csv", index=False, encoding="utf-8-sig")
    year_counts(frames, front, start, end).to_csv(
        out / "yearly_coverage.csv", index=False, encoding="utf-8-sig")
    markets = []
    for name, data in frames.items():
        markets.append({"market": name,
            "earliest": str(data.index.min().date()) if data is not None else None,
            "latest": str(data.index.max().date()) if data is not None else None,
            "bars": len(data) if data is not None else 0})
    full = front[front.complete] if not front.empty else pd.DataFrame()
    markets.append({"market": "TX_FULL",
        "earliest": str(full.date.min().date()) if len(full) else None,
        "latest": str(full.date.max().date()) if len(full) else None,
        "bars": len(full)})
    pd.DataFrame(markets).to_csv(out / "market_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["source", "start", "end", "error"]).to_csv(
        out / "source_failures.csv", index=False, encoding="utf-8-sig")
    aligned_rows = 0
    if all(frames.get(m) is not None for m in SYMBOLS):
        aligned = cross_market_asof(frames["TAIEX"].index,
                                    frames["NASDAQ"].index, frames["SOX"].index)
        aligned.to_csv(out / "cross_market_asof.csv", index=False, encoding="utf-8-sig")
        aligned_rows = len(aligned)
    complete = (all(frames.get(m) is not None and len(frames[m]) >= 200 for m in SYMBOLS)
                and len(full) >= 200 and not errors and
                quality["duplicate_rows"] == 0 and quality["invalid_rows"] == 0)
    status = {"version": "V3.5-A", "success": True, "complete": bool(complete),
        "requested_start": str(start), "requested_end": str(end),
        "markets": markets, "tx_quality": quality,
        "first_complete_tx_day": str(full.date.min().date()) if len(full) else None,
        "source_failures": len(errors), "tw_decision_cutoffs": aligned_rows,
        "elapsed_seconds": round(time.monotonic() - begun, 2)}
    (out / "audit_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Kronos V3.5-A 四市場歷史資料稽核", "",
        f"查詢：{start} ～ {end}", f"來源與 OHLC 初步驗證通過：{complete}", "",
        "| 市場 | 最早取得日 | 最新取得日 | 有效筆數 |",
        "|---|---|---|---:|"]
    for row in markets:
        lines.append(f"| {row['market']} | {row['earliest'] or '未取得'} | "
                     f"{row['latest'] or '未取得'} | {row['bars']} |")
    lines.extend(["", f"TX 最早完整日夜盤交易日：{status['first_complete_tx_day'] or '未取得'}",
        f"TX 無效/重複紀錄：{quality['invalid_rows']}/{quality['duplicate_rows']}",
        f"失敗下載區段：{len(errors)}", "",
        "注意：本階段不做模型推論；最早日不代表十年期間所有交易日齊全。",
        "參照 yearly_coverage.csv、tx_front_sessions.csv、source_failures.csv。",
        "cross_market_asof.csv 包含台灣 17:10 與 06:20 當時可用的美股日 K。",
        "夜盤採下一個一般交易日之歸屬日期；不得用台美相同曆日直接合併。",
        "後續仍需驗證換月、假期、資料品質與預訓練污染。"])
    (out / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"FINISHED complete={complete} errors={len(errors)} TX_full={len(full)}")
    return status


def main():
    parser = argparse.ArgumentParser(description="V3.5 isolated 4-market historical data audit")
    parser.add_argument("--start", default="2017-05-16")
    parser.add_argument("--end", default=datetime.now(TPE).strftime("%Y-%m-%d"))
    parser.add_argument("--out", default="outputs_v35/audit")
    parser.add_argument("--tx-chunk-days", type=int, default=179)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start > end:
        parser.error("start > end")
    run(start, end, args.out, chunk_days=args.tx_chunk_days)


if __name__ == "__main__":
    main()
