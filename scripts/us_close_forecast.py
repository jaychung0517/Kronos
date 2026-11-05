"""V3.4 US-close run: US forecasts + latest archived Taiwan close comparison.

Place as scripts/us_close_forecast.py in Kronos repo. Read-only, no orders.
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import torch
import yfinance as yf

from model import Kronos, KronosTokenizer, KronosPredictor
from scripts import taiwan_forecast as tw

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
NY = ZoneInfo("America/New_York")
TPE = ZoneInfo("Asia/Taipei")
US = {"NASDAQ": "^IXIC", "SOX": "^SOX"}
ART_TW = "kronos-taiwan-v3-results"
ART_US = "kronos-us-close-results"


def log(msg):
    print(f"[US CLOSE] {msg}", flush=True)


def get_us_history(symbol: str, expected_date):
    df = yf.download(symbol, period="3y", interval="1d", auto_adjust=False,
                     progress=False, multi_level_index=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo empty data: {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = df[tw.OHLC].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY).tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 200:
        raise RuntimeError(f"{symbol}: fewer than 200 bars")
    tw.validate_ohlc(df, symbol)
    actual = df.index[-1].date()
    if actual != expected_date:
        raise ValueError(f"{symbol}: latest complete US session is {actual}, expected {expected_date}; do not use stale/partial data")
    return df


def future_us_dates(last_date):
    calendar = mcal.get_calendar("NYSE")
    dates = calendar.valid_days(start_date=pd.Timestamp(last_date) + pd.Timedelta(days=1),
                                end_date=pd.Timestamp(last_date) + pd.Timedelta(days=35))
    if len(dates) < 5:
        raise RuntimeError("US calendar lacks next five sessions")
    return pd.DatetimeIndex(dates[:5]).tz_localize(None)


def github_archived_csv(artifact_name, target_file, max_age_days=8):
    """Best-effort download from latest successful workflow artifact.

    Artifact might be absent/expired; caller must mark missing, not invent data.
    """
    repo, token = os.getenv("GITHUB_REPOSITORY", ""), os.getenv("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    root = f"https://api.github.com/repos/{repo}"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        runs = requests.get(f"{root}/actions/workflows/taiwan_kronos.yml/runs",
                            params={"status": "success", "per_page": 30},
                            headers=headers, timeout=30)
        runs.raise_for_status()
        for run in runs.json().get("workflow_runs", []):
            if run.get("event") not in ("schedule", "workflow_dispatch"):
                continue
            created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
            if (datetime.now(created.tzinfo) - created).days > max_age_days:
                continue
            response = requests.get(f"{root}/actions/runs/{run['id']}/artifacts",
                                    headers=headers, timeout=30)
            response.raise_for_status()
            for artifact in response.json().get("artifacts", []):
                if artifact.get("expired") or artifact.get("name") != artifact_name:
                    continue
                download = requests.get(artifact["archive_download_url"],
                                        headers=headers, timeout=90)
                download.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
                    names = archive.namelist()
                    matches = [x for x in names if x == target_file or x.endswith("/" + target_file)]
                    if matches:
                        return pd.read_csv(io.BytesIO(archive.read(matches[0])))
    except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
        log(f"Historical comparison unavailable: {type(exc).__name__}")
    return None


def mk_summary(market, frame, pred_list, us_session):
    r = np.array([p["return_pct"].to_numpy(dtype=float) for p in pred_list])
    path = np.mean(r, axis=0)
    votes = tw.v33_ai_votes(r[:, 2], r[:, 0], r[:, 4])
    price = tw.v33_price_features(frame)
    decision = tw.v33_combine(price, votes)
    return {
        "market": market, "origin": str(us_session), "close": float(frame.close.iloc[-1]),
        "price_state": decision["signal"],
        "votes_up_3d": votes["up_votes_3d"], "votes_down_3d": votes["down_votes_3d"],
        "valid_seeds": votes["valid_seeds"],
        "pred_1d_pct": float(path[0]), "pred_3d_pct": float(path[2]),
        "pred_5d_pct": float(path[4]),
        "valid_ohlc_rate": float(np.mean([p["ohlc_valid"].all() for p in pred_list])),
    }, path


def direction(value):
    v = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(v):
        return "UNKNOWN"
    return "UP" if v > 0 else "DOWN" if v < 0 else "NEUTRAL"


def write_dashboard(us_data, tw_data, prior_us, session_date):
    labels = {"TAIEX": "加權指數", "TX": "台指期", "NASDAQ": "那斯達克", "SOX": "費城半導體"}
    all_rows = []
    if tw_data is not None:
        for _, row in tw_data.iterrows():
            market = row.get("market")
            if market not in ("TAIEX", "TX"):
                continue
            stamp = str(row.get("origin", ""))
            age = (datetime.now(TPE).date() - pd.to_datetime(stamp).date()).days
            if not 0 <= age <= 4:
                log(f"Archived Taiwan signal too old: {market} {stamp}")
                continue
            all_rows.append({"market": market, "asof": stamp,
                             "state": str(row.get("signal", "UNKNOWN")),
                             "return_3d": float(row["forecast_3d_pct"]),
                             "seed_up": int(row["up_votes_3d"]), "seed_count": int(row["valid_seeds"]),
                             "fresh": False})
    for market in US:
        s = us_data[market]["summary"]
        all_rows.append({"market": market, "asof": str(session_date),
                         "state": s["price_state"], "return_3d": s["pred_3d_pct"],
                         "seed_up": s["votes_up_3d"], "seed_count": s["valid_seeds"],
                         "fresh": True})
    for row in all_rows:
        row["direction_3d"] = direction(row["return_3d"])

    table = pd.DataFrame(all_rows)
    table.to_csv(OUT / "us_close_four_market.csv", index=False, encoding="utf-8-sig")
    # Agreement is raw direction only. Cross-market sessions and model errors are correlated.
    full = len(table) == 4 and set(table["market"]) == {"TAIEX", "TX", "NASDAQ", "SOX"}
    agree = bool(full and table.direction_3d.nunique() == 1)
    consensus = ("4/4 同方向（尚未驗證準確率）" if agree else
                 "方向分歧" if full else "台股快照缺漏：不計算四市場一致")
    previous = {}
    if prior_us is not None and "market" in prior_us.columns:
        for _, row in prior_us.iterrows():
            if row.get("market") in US:
                previous[row["market"]] = row.get("pred_3d_pct")

    fig, axes = plt.subplots(3, 1, figsize=(12.4, 11.3),
                             gridspec_kw={"height_ratios": [2, 2, 1.6]})
    try:
        from matplotlib import font_manager
        paths = [p for p in font_manager.findSystemFonts() if "NotoSansCJK" in p or "NotoSansCJK" in p.replace("-", "")]
        if paths:
            font_manager.fontManager.addfont(paths[0])
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=paths[0]).get_name()
    except Exception:
        pass
    for ax, (market, item) in zip(axes[:2], us_data.items()):
        history = item["history"].close.tail(45)
        forecast = item["forecast"]
        dates = item["dates"]
        ax.plot(history.index, history.values, label="History", color="#3568a7")
        ax.plot([history.index[-1], *dates], [history.iloc[-1], *forecast],
                "o--", label="US forecast (3-seed mean)", color="#d48a22")
        s = item["summary"]
        old = previous.get(market)
        delta = "" if old is None or not np.isfinite(pd.to_numeric(old, errors="coerce")) else f" | vs last US forecast {s['pred_3d_pct'] - float(old):+.2f}pp"
        ax.set_title(f"{market}  {session_date}  D3: {s['pred_3d_pct']:+.2f}%  up {s['votes_up_3d']}/{s['valid_seeds']}{delta}", fontsize=11)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=9)
    ax = axes[2]
    ax.axis("off")
    ax.text(.01, .95, f"US CLOSE {session_date} | {consensus}", fontsize=13, weight="bold", transform=ax.transAxes, va="top")
    for i, row in enumerate(all_rows):
        label = labels.get(row["market"], row["market"])
        txt = f"{label:8s} {row['direction_3d']:7s}  D3 {row['return_3d']:+6.2f}% | votes {row['seed_up']}/{row['seed_count']} | as-of {row['asof']}"
        ax.text(.02, .78 - .18*i, txt, fontsize=11, transform=ax.transAxes, va="top")
    ax.text(.02, .05, "Taiwan signals are archived TW-close predictions, not recalculated from US close. No trading orders.", fontsize=9, transform=ax.transAxes)
    fig.tight_layout(pad=2.2)
    png = OUT / "us_close_dashboard.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    lines = [f"# {session_date} 美股收盤後跨市場觀察", "", f"一致性：{consensus}",
             "", "| 市場 | 資料日期 | 三日預測 | 看多票 | 方向 |", "|---|---|---:|---:|---|"]
    for row in all_rows:
        lines.append(f"| {row['market']} | {row['asof']} | {row['return_3d']:+.2f}% | {row['seed_up']}/{row['seed_count']} | {row['direction_3d']} |")
    lines += ["", "台股欄位引用前一份台股收盤報告；不能當作美股收盤後的即時台股模型更新。",
              "美股與台股三日交易日視窗不完全相同，四市場一致不是經驗證的勝率或機率。",
              "新預測與上次美股預測差距是不同起點的預測變化，不是當日預測誤差。"]
    (OUT / "us_close_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return consensus, table


def send_photo(date, consensus):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        log("Telegram secrets missing; PNG/CSV stored as artifact")
        return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with (OUT / "us_close_dashboard.png").open("rb") as fh:
        resp = requests.post(url, data={"chat_id": chat,
                  "caption": f"🌅 {date} 美股收盤後台股早盤觀察\n{consensus}\n⚠️ 研究訊號，非交易建議。"},
                  files={"photo": ("us_close_dashboard.png", fh, "image/png")}, timeout=50)
    # Never print the Telegram URL containing the token.
    if resp.status_code != 200 or not resp.json().get("ok", False):
        raise RuntimeError(f"Telegram sendPhoto rejected, HTTP {resp.status_code}")
    log("Telegram image sent")


def main():
    now = datetime.now(NY)
    status = {"version": "V3.4-US-close", "created_at_taipei": datetime.now(TPE).isoformat(),
              "success": False, "mode": "us_close", "us_market_date": str(now.date())}
    try:
        if now.weekday() >= 5 or (now.hour, now.minute) < (16, 30):
            status.update({"skipped": True, "reason": "US regular session not closed / weekend"})
            return
        histories = {name: get_us_history(symbol, now.date()) for name, symbol in US.items()}
        dates = future_us_dates(now.date())
        log("US regular close verified for both indices")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained(os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-base"))
        tokenizer.eval(); model.eval()
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        us_data = {}
        all_paths = []
        for market, frame in histories.items():
            predictions = []
            for i in range(3):
                seed = tw.experiment_seed(market, frame.index[-1], i) + (10000 if market == "NASDAQ" else 20000)
                pred = tw.predict(predictor, frame.tail(200), dates, sample_count=5, seed=seed)
                predictions.append(pred)
                for j, (date, row) in enumerate(pred.iterrows(), start=1):
                    all_paths.append({"market": market, "seed_index": i, "target": str(date.date()),
                                      "horizon": j, "return_pct": float(row.return_pct),
                                      "ohlc_valid": bool(row.ohlc_valid)})
            summary, path = mk_summary(market, frame, predictions, now.date())
            forecast = float(frame.close.iloc[-1]) * (1 + path/100)
            us_data[market] = {"history": frame, "dates": dates, "summary": summary, "forecast": forecast}
            log(f"{market}: D1={summary['pred_1d_pct']:+.2f}% D3={summary['pred_3d_pct']:+.2f}% votes {summary['votes_up_3d']}/3")
        pd.DataFrame(all_paths).to_csv(OUT / "us_close_seed_paths.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([i["summary"] for i in us_data.values()]).to_csv(
            OUT / "us_close_summary.csv", index=False, encoding="utf-8-sig")
        tw_data = github_archived_csv(ART_TW, "v33_reversal_live.csv")
        prior_us = github_archived_csv(ART_US, "us_close_summary.csv")
        consensus, table = write_dashboard(us_data, tw_data, prior_us, now.date())
        status.update({"success": True, "consensus": consensus, "market_count": len(table), "tw_snapshot_available": tw_data is not None})
        if os.getenv("REPORT_TELEGRAM", "1") == "1":
            try:
                send_photo(now.date(), consensus)
            except (RuntimeError, requests.RequestException, ValueError) as exc:
                status["telegram_error"] = f"{type(exc).__name__}: {exc}"
                log("Telegram send failed; report remains in Actions artifact")
    except ValueError as exc:
        # Stale US bars / US market holiday: no misleading report.
        status.update({"skipped": True, "reason": str(exc)})
        log(f"Skipped: {exc}")
    except Exception as exc:
        status.update({"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        (OUT / "us_close_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
