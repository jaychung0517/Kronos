"""Kronos V3.5-B: isolated four-market rolling-direction backtest.

Consumes an immutable V3.5-A data-audit artifact and does not import
production forecast scripts or touch scheduled live predictions.

Each rank is one historical Taiwan-market origin. At TW_CLOSE, use the
latest US close already known. At the following US_CLOSE (when one
exists before next Taiwan opening), reuse Taiwan bars and update US bars.
"""
from __future__ import annotations
import argparse
from datetime import datetime, time as clock, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.kronos_v35_data_audit import OHLC, valid_ohlc_rows

TPE = ZoneInfo("Asia/Taipei")
NY = ZoneInfo("America/New_York")
SEEDS = 3
H = 5
MARKETS = ("TAIEX", "TX", "NASDAQ", "SOX")
HORIZONS = (1, 3, 5)
BASE_SEED = 20261104
LABELS = {
    "TAIEX": 0, "TX": 1000, "NASDAQ": 10000, "SOX": 20000
}


def log(message):
    print("[V35 BACKTEST] " + message, flush=True)


def valid_input(df, context):
    if len(df) != 200 or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{context}: input length, dates or order invalid")
    if not valid_ohlc_rows(df).all():
        raise ValueError(f"{context}: invalid history OHLC")


def read_snapshot(folder):
    root = Path(folder)
    frames = {}
    for market in ("TAIEX", "NASDAQ", "SOX"):
        filename = root / f"{market.lower()}_ohlc.csv.gz"
        df = pd.read_csv(filename, index_col=0, parse_dates=[0])
        if not set(OHLC).issubset(df.columns):
            raise ValueError(f"{market}: missing OHLC")
        df = df[OHLC].sort_index()
        if not valid_ohlc_rows(df).all() or df.index.has_duplicates:
            raise ValueError(f"{market}: invalid audit snapshot")
        frames[market] = df
    tx = pd.read_csv(root / "tx_valid_sessions.csv.gz", parse_dates=["date"])
    required = {"date", "contract_date", "trading_session",
                "open", "max", "min", "close", "volume"}
    if not required.issubset(tx.columns) or tx.empty:
        raise ValueError("TX snapshot missing required rows/columns")
    tx["contract_date"] = tx.contract_date.astype(str)
    return frames, tx


def make_contract_bars(tx):
    rows = []
    for (day, contract), group in tx.groupby(["date", "contract_date"], sort=True):
        if group.trading_session.duplicated().any():
            raise ValueError(f"duplicated TX session {day}/{contract}")
        daypart = group.loc[group.trading_session.eq("position")]
        nightpart = group.loc[group.trading_session.eq("after_market")]
        if len(daypart) != 1:
            continue
        d = daypart.iloc[0]
        n = nightpart.iloc[0] if len(nightpart) == 1 else None
        parts = [d] if n is None else [n, d]
        rows.append({"date": pd.Timestamp(day), "contract_date":str(contract),
            "open": float(parts[0]["open"]), "high": max(float(p["max"]) for p in parts),
            "low": min(float(p["min"]) for p in parts), "close":float(d["close"]),
            "volume": sum(float(p["volume"]) for p in parts),
            "has_night": n is not None})
    bars = pd.DataFrame(rows).sort_values(["date", "contract_date"])
    if bars.empty or bars.duplicated(["date", "contract_date"]).any():
        raise ValueError("TX contract bars invalid")
    if not valid_ohlc_rows(bars).all():
        raise ValueError("TX contract OHLC invalid")
    return bars


def make_front(bars):
    active = bars[bars.contract_date >= bars.date.dt.strftime("%Y%m")].copy()
    allfront = active.sort_values(["date", "contract_date"]).drop_duplicates("date")
    # Never substitute next-month bars when selected front contract lacks night data.
    complete = allfront.loc[allfront.has_night].set_index("date").sort_index()
    if len(complete) < 2005:
        raise ValueError(f"Only {len(complete)} complete TX front-month dates")
    return complete, bars


def tx_asof(full_front, bars, origin, window=200):
    frame = full_front.loc[:origin].tail(window).copy().reset_index()
    if len(frame) != window:
        raise ValueError(f"TX insufficient history at {origin}")
    quotes = bars.loc[bars.date <= origin].pivot(
        index="date", columns="contract_date", values="close").sort_index()
    for i in range(1, len(frame)):
        old, new = str(frame.at[i - 1, "contract_date"]), str(frame.at[i, "contract_date"])
        if old == new:
            continue
        if old not in quotes or new not in quotes or new < old:
            raise ValueError(f"TX roll contracts invalid {origin}: {old}->{new}")
        overlap = quotes.loc[:frame.at[i, "date"], [old, new]].dropna()
        if overlap.empty or (frame.at[i, "date"] - overlap.index[-1]).days > 10:
            raise ValueError(f"TX no as-of overlap for {old}->{new} at {origin}")
        gap = float(overlap.iloc[-1][new] - overlap.iloc[-1][old])
        frame.loc[:i-1, OHLC] = frame.loc[:i-1, OHLC].add(gap)
    result = frame.set_index("date")[OHLC].sort_index()
    if not valid_ohlc_rows(result).all():
        raise ValueError(f"TX back-adjusted invalid at {origin}")
    return result


def close_time(market, d):
    d = pd.Timestamp(d).date()
    if market in ("TAIEX", "TX"):
        wall = clock(13, 30) if market == "TAIEX" else clock(13, 45)
        return datetime.combine(d, wall, tzinfo=TPE)
    return datetime.combine(d, clock(16), tzinfo=NY).astimezone(TPE)


def last_market_date(market, available_dates, cutoff):
    return next((d for d in reversed(available_dates)
                 if close_time(market, d) <= cutoff), None)


def session_times(tw_day, next_tw_day, us_dates):
    twclose = datetime.combine(pd.Timestamp(tw_day).date(), clock(17, 10), tzinfo=TPE)
    nextopen = datetime.combine(pd.Timestamp(next_tw_day).date(), clock(8, 45), tzinfo=TPE)
    result = [("TW_CLOSE", twclose)]
    # US_CLOSE at 06:20 Taipei after a *new* US session and before next TW open.
    newest = [close_time("NASDAQ", d) for d in us_dates
              if twclose < close_time("NASDAQ", d) < nextopen]
    if newest:
        usclose = max(newest)
        morning = datetime.combine(usclose.date(), clock(6, 20), tzinfo=TPE)
        if usclose <= morning < nextopen:
            result.append(("US_CLOSE", morning))
    return result


def seed_for(market, origin, index):
    return (BASE_SEED + int(pd.Timestamp(origin).strftime("%Y%m%d")) +
            LABELS[market] + index * 100003)


def predict(predictor, market, origin, history, dates, seed):
    valid_input(history, f"{market} {origin}")
    if len(dates) != H:
        raise ValueError("Expected five future market dates")
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    with torch.inference_mode():
        raw = predictor.predict(
            df=history[OHLC].reset_index(drop=True),
            x_timestamp=pd.Series(pd.DatetimeIndex(history.index)),
            y_timestamp=pd.Series(pd.DatetimeIndex(dates)),
            pred_len=H, T=1.0, top_p=0.9, sample_count=5, verbose=False)
    if not isinstance(raw, pd.DataFrame) or len(raw) != H or not set(OHLC).issubset(raw.columns):
        raise ValueError("Kronos generated invalid shape")
    result = raw[OHLC].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    close = result.close.to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Kronos generated invalid close")
    price_ok = valid_ohlc_rows(result).to_numpy()
    reference = float(history.close.iloc[-1])
    return (100.0 * (close / reference - 1.0), price_ok)


def get_history(market, origin, frames, front, bars):
    if market == "TX":
        return tx_asof(front, bars, origin, 200)
    history = frames[market].loc[:origin].tail(200)
    valid_input(history, f"{market} {origin}")
    return history


def market_future(market, origin, frames, front):
    if market == "TX":
        # TX and TAIEX must describe the *same* coming five Taiwan sessions.
        # If any TX day/night is missing on a stock-market date, skip the
        # affected origin rather than quietly shifting the TX forecast horizon.
        dates = frames["TAIEX"].index
        future = dates[dates > origin][:H]
        if len(future) != H or not future.isin(front.index).all():
            raise ValueError(f"TX: missing full day/night among next five TW sessions after {origin}")
        return future
    dates = frames[market].index
    after = dates[dates > origin][:H]
    if len(after) < H:
        raise ValueError(f"{market}: missing future sessions after {origin}")
    return after


def realized_return(market, origin, target, frames, front, bars):
    if market == "TX":
        # Score using adjustment observable at *target*, never as predictor input.
        current = tx_asof(front, bars, target, 200)
        return 100.0 * (float(current.at[target, "close"]) /
                        float(current.at[origin, "close"]) - 1.0)
    df = frames[market]
    return 100.0 * (float(df.at[target, "close"]) / float(df.at[origin, "close"]) - 1.0)


def eligible_origins(frames, front):
    tw_dates = frames["TAIEX"].index
    common = tw_dates.intersection(front.index).sort_values()
    if len(common) < 2205:
        raise ValueError(f"Too few TAIEX/TX common sessions for 2000 tests: {len(common)}")
    candidates = []
    for origin in common[199:]:
        following = tw_dates[tw_dates > origin][:H]
        if len(following) == H and following.isin(front.index).all():
            candidates.append(origin)
    # Fixed rank ordering across every 100/500/1000/2000 stage.
    # Source OHLC is still truncated independently at each market origin.
    return list(reversed(candidates))


def run(snapshot, rank_start, rank_count, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, tx = read_snapshot(snapshot)
    bars = make_contract_bars(tx)
    front, bars = make_front(bars)
    origins = eligible_origins(frames, front)
    if len(origins) < 2000:
        raise ValueError(f"Only {len(origins)} eligible origins; 2000 required")
    subset = origins[rank_start:rank_start + rank_count]
    if len(subset) != rank_count:
        raise ValueError(f"Rank slice {rank_start}:{rank_start + rank_count} incomplete")
    log(f"Worker origins ranks={rank_start}..{rank_start + rank_count-1}, "
        f"first={subset[0].date()}, last={subset[-1].date()}")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    results = []
    errors = []
    cache = {}
    last_log = 0
    # For the US_CLOSE daily report, TW snapshots remain at previous TW close;
    # freshly completed US sessions are included by timestamp. Same 4 indicators.
    for i, tw_origin in enumerate(subset):
        tw_pos = frames["TAIEX"].index.get_loc(tw_origin)
        next_tw = frames["TAIEX"].index[tw_pos + 1]
        sessions = session_times(tw_origin, next_tw, frames["NASDAQ"].index)
        for session, cutoff in sessions:
            for market in MARKETS:
                local_dates = front.index if market == "TX" else frames[market].index
                market_origin = last_market_date(market, local_dates, cutoff)
                if market_origin is None:
                    raise ValueError(f"{market} no data as-of {cutoff}")
                if market in ("TAIEX", "TX") and market_origin != tw_origin:
                    raise ValueError(f"{market} TW cutoff misaligned {market_origin}!={tw_origin}")
                if close_time(market, market_origin) > cutoff:
                    raise ValueError(f"LOOKAHEAD {market} at {cutoff}")
                target_dates = market_future(market, market_origin, frames, front)
                key = (market, market_origin)
                if key not in cache:
                    hist = get_history(market, market_origin, frames, front, bars)
                    cache[key] = []
                    for seed_index in range(SEEDS):
                        seed = seed_for(market, market_origin, seed_index)
                        try:
                            path, quality = predict(
                                predictor, market, market_origin, hist, target_dates, seed)
                            cache[key].append((seed, path, quality))
                        except Exception as exc:
                            errors.append({"tw_origin": str(tw_origin.date()), "market": market,
                                           "market_origin": str(market_origin.date()),
                                           "seed_index": seed_index, "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
                            raise
                predictions = cache[key]
                actuals = {h: realized_return(
                    market, market_origin, target_dates[h - 1],
                    frames, front, bars) for h in HORIZONS}
                tw_futures = market_future("TAIEX", tw_origin, frames, front)
                tw_actual = {h: realized_return(
                    "TAIEX", tw_origin, tw_futures[h - 1], frames, front, bars)
                    for h in HORIZONS}
                for seed_index, (seed, path, quality) in enumerate(predictions):
                    for h in HORIZONS:
                        results.append({
                            "rank": rank_start + i,
                            "tw_origin": str(tw_origin.date()),
                            "session": session,
                            "decision_time_tpe": cutoff.isoformat(),
                            "market": market,
                            "model_origin": str(market_origin.date()),
                            "input_cutoff_tpe": close_time(market, market_origin).isoformat(),
                            "horizon": h,
                            "seed_index": seed_index,
                            "random_seed": seed,
                            "pred_return_pct": float(path[h - 1]),
                            "realized_market_return_pct": float(actuals[h]),
                            "realized_taiex_return_pct": float(tw_actual[h]),
                            "path_ohlc_valid": bool(quality.all()),
                            "target_date": str(target_dates[h - 1].date()),
                        })
        # Store progress after each ten origins to protect partial diagnostic output.
        if (i + 1) % 10 == 0 or i + 1 == len(subset):
            pd.DataFrame(results).to_csv(output / f"forecast_shard_{rank_start:04d}.csv",
                                         index=False, encoding="utf-8-sig")
            log(f"progress {i + 1}/{len(subset)}; rows={len(results)}")
    if errors:
        pd.DataFrame(errors).to_csv(output / f"errors_{rank_start:04d}.csv", index=False)
        raise RuntimeError(f"Model failed on {len(errors)} forecasts")
    log(f"SHARD COMPLETE rank={rank_start}, origins={len(subset)}, records={len(results)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="outputs_v35/audit/market_cache")
    p.add_argument("--rank-start", type=int, required=True)
    p.add_argument("--rank-count", type=int, default=100)
    p.add_argument("--output", default="outputs_v35/shards")
    args = p.parse_args()
    if args.rank_start < 0 or args.rank_count <= 0 or args.rank_start + args.rank_count > 2000:
        p.error("Invalid rank slice")
    run(args.snapshot, args.rank_start, args.rank_count, args.output)


if __name__ == "__main__":
    main()
