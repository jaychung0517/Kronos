"""Kronos V3.3 daily one-page report + optional Telegram photo.

Run after `python -m scripts.taiwan_forecast`, in the repository root:
    python -m scripts.kronos_daily_report

Reads only existing outputs. Does not change forecasts or trading signals.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import requests

OUT = Path("outputs")
IMAGE = OUT / "kronos_daily_dashboard.png"
REPORT = OUT / "kronos_daily_dashboard.md"
MARKETS = ("TAIEX", "TX")

STATUS_ZH = {
    "TOP_WATCH": "高檔轉弱觀察",
    "TOP_CONFIRM": "高檔轉弱（價格條件）",
    "BOTTOM_WATCH": "急跌後止跌觀察",
    "BOTTOM_CONFIRM": "低檔反轉（價格條件）",
    "TREND_UP": "趨勢偏多",
    "TREND_DOWN_OR_RANGE": "下跌或區間整理",
}

def setup_chinese_font():
    preferred = ("Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft JhengHei")
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[REPORT] WARNING: CJK font not found; Chinese may display as boxes", flush=True)
    plt.rcParams["axes.unicode_minus"] = False


def read_history(market: str) -> pd.Series:
    file = OUT / ("taiex_history.csv" if market == "TAIEX" else "tx_continuous_adjusted.csv")
    df = pd.read_csv(file)
    date_col = "Date" if market == "TAIEX" else "date"
    if date_col not in df or "close" not in df:
        raise ValueError(f"{file}: missing {date_col}/close")
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df["close"] = pd.to_numeric(df["close"], errors="raise")
    if df[date_col].duplicated().any() or df["close"].isna().any():
        raise ValueError(f"{file}: duplicate dates or missing close")
    series = df.set_index(date_col)["close"].sort_index()
    if len(series) < 25:
        raise ValueError(f"{file}: too little history")
    return series


def read_inputs():
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if not status.get("success") or status.get("mode") != "live":
        raise ValueError("Daily report only accepts successful LIVE forecasts")

    states = pd.read_csv(OUT / "v33_reversal_live.csv").set_index("market")
    paths = pd.read_csv(OUT / "v33_seed_paths.csv")
    paths["target_date"] = pd.to_datetime(paths["target_date"], errors="raise")
    paths["origin"] = pd.to_datetime(paths["origin"], errors="raise")

    if set(states.index) != set(MARKETS):
        raise ValueError("v33_reversal_live.csv must contain TAIEX and TX")
    if paths.duplicated(["market", "seed_index", "horizon"]).any():
        raise ValueError("Duplicate market/seed/horizon forecast")

    summaries = {}
    origin_dates = set()
    for market in MARKETS:
        row = states.loc[market]
        history = read_history(market)
        origin = pd.Timestamp(row["origin"]).normalize()
        origin_dates.add(origin)
        if history.index[-1].normalize() != origin:
            raise ValueError(f"{market}: history date does not match signal date")
        sub = paths.loc[paths["market"] == market].copy()
        if sub.empty or set(sub["horizon"].astype(int)) != set(range(1, 6)):
            raise ValueError(f"{market}: missing D1-D5 predictions")
        if set(sub["origin"].dt.normalize()) != {origin}:
            raise ValueError(f"{market}: forecast origin inconsistent")
        count = sub.groupby("seed_index")["horizon"].nunique()
        if (count != 5).any() or len(count) != int(row["valid_seeds"]):
            raise ValueError(f"{market}: missing seeds or daily paths")

        sub["pred_return_pct"] = pd.to_numeric(sub["pred_return_pct"], errors="raise")
        if not np.isfinite(sub["pred_return_pct"]).all():
            raise ValueError(f"{market}: non-finite forecast")

        grouped = sub.groupby("horizon", sort=True)
        pred = grouped.agg(
            target_date=("target_date", "first"),
            mean_ret=("pred_return_pct", "mean"),
            min_ret=("pred_return_pct", "min"),
            max_ret=("pred_return_pct", "max"),
        )
        if pred["target_date"].duplicated().any() or not pred["target_date"].is_monotonic_increasing:
            raise ValueError(f"{market}: invalid target dates")
        last = float(history.iloc[-1])
        for col in ("mean_ret", "min_ret", "max_ret"):
            pred[col.replace("_ret", "_close")] = last * (1 + pred[col] / 100)
        summaries[market] = (history.tail(55), row, pred)

    if len(origin_dates) != 1:
        raise ValueError("TAIEX and TX use different origin dates")
    if "latest_data_date" in status:
        if str(next(iter(origin_dates)).date()) != str(status["latest_data_date"]):
            raise ValueError("status date inconsistent with market origin")
    return next(iter(origin_dates)), summaries


def make_report(origin, summaries):
    setup_chinese_font()
    fig = plt.figure(figsize=(10, 12.4), facecolor="#f8fafc")
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 3.25, 3.25, 1.25], hspace=0.34,
                          left=0.09, right=0.96, top=0.975, bottom=0.045)
    heading = fig.add_subplot(gs[0]); heading.axis("off")
    heading.text(0, 0.83, "Kronos V3.3｜台灣大盤方向與反轉觀察", fontsize=19, weight="bold", color="#0f172a")
    heading.text(0, 0.19, f"資料截止：{origin:%Y/%m/%d} 收盤  ·  B組 200K × 5抽樣 × 3種子", fontsize=10.5, color="#475569")
    text_rows = [f"# Kronos V3.3 每日市場報告｜{origin:%Y-%m-%d}", "", "以下為模型研究與價格條件觀察，不是交易指令。", ""]

    for i, market in enumerate(MARKETS):
        history, row, pred = summaries[market]
        ax = fig.add_subplot(gs[i + 1])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        forecast = pd.concat([
            pd.Series([history.iloc[-1]], index=[history.index[-1]]),
            pd.Series(pred["mean_close"].to_numpy(), index=pd.DatetimeIndex(pred["target_date"]))
        ])
        ax.plot(history.index, history.to_numpy(), color="#2563eb", linewidth=2.1, label="歷史收盤")
        ax.plot(forecast.index, forecast.to_numpy(), color="#f97316", linewidth=2.4, linestyle="--", marker="o", markersize=5, label="三種子平均預測")
        ax.fill_between(pred["target_date"].to_numpy(), pred["min_close"].to_numpy(), pred["max_close"].to_numpy(), color="#f97316", alpha=0.13, label="種子預測範圍")
        ax.axvline(history.index[-1], color="#64748b", linewidth=1, linestyle=":")
        ax.grid(True, alpha=0.16)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(labelsize=9)
        ax.set_title(f"{'加權指數 TAIEX' if market == 'TAIEX' else '台指期 TX'}｜{STATUS_ZH.get(str(row['signal']), str(row['signal']))}",
                     loc="left", fontsize=13.5, color="#0f172a", weight="bold", pad=12)
        votes = f"{int(row['up_votes_3d'])}/{int(row['valid_seeds'])} 看多"
        confirmed = "是" if bool(row.get("top_confirmed", False) or row.get("bottom_confirmed", False)) else "否"
        line = (f"現價 {float(row['close']):,.2f}   當日 {float(row['return_1d_pct']):+.2f}%   "
                f"近5日 {float(row['return_5d_pct']):+.2f}%\n"
                f"AI 3日 {votes}   D1 {pred.loc[1,'mean_ret']:+.2f}%   "
                f"D3 {pred.loc[3,'mean_ret']:+.2f}%   D5 {pred.loc[5,'mean_ret']:+.2f}%   "
                f"價格確認：{confirmed}")
        ax.text(0.02, 0.04, line, transform=ax.transAxes, fontsize=9.4, va="bottom", color="#0f172a",
                bbox={"boxstyle":"round,pad=0.65", "facecolor":"white", "edgecolor":"#e2e8f0", "alpha":0.93})
        ax.legend(loc="upper right", fontsize=8.7, frameon=True)
        text_rows += [f"## {market}｜{STATUS_ZH.get(str(row['signal']),str(row['signal']))}",
                      f"- 收盤：{float(row['close']):,.2f}；當日：{float(row['return_1d_pct']):+.2f}%；最近5日：{float(row['return_5d_pct']):+.2f}%",
                      f"- 1/3/5日三種子平均預測：{pred.loc[1,'mean_ret']:+.2f}% / {pred.loc[3,'mean_ret']:+.2f}% / {pred.loc[5,'mean_ret']:+.2f}%",
                      f"- AI三日看多：{votes}；價格確認：{confirmed}", ""]

    footer = fig.add_subplot(gs[3]); footer.axis("off")
    footer.text(0.01, 0.83, "判讀重點", fontsize=13, weight="bold", color="#0f172a")
    footer.text(0.01, 0.58, "● 觀察訊號 ≠ 底部／頂部確認。高檔檢查停利，急跌後觀察止跌及個股買點。", fontsize=10.2, color="#334155")
    footer.text(0.01, 0.35, "● 橘線：3種子平均。淡橘區間：不同種子預測範圍，非信賴區間。", fontsize=10.2, color="#334155")
    footer.text(0.01, 0.12, "● AI一致性不是上漲機率；模型歷史方向表現尚未證實具交易優勢。", fontsize=10.2, color="#64748b")
    fig.savefig(IMAGE, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    text_rows += ["---", "多種子一致性不是市場上漲機率；價格未確認反轉前，不將訊號視為自動買賣建議。", ""]
    REPORT.write_text("\n".join(text_rows), encoding="utf-8")
    return "\n".join(text_rows)


def send_telegram(origin, summaries):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[REPORT] Telegram Secrets missing; image created, notification skipped", flush=True)
        return False
    labels = []
    for market in MARKETS:
        _, row, pred = summaries[market]
        labels.append(f"{market}：{STATUS_ZH.get(str(row['signal']), row['signal'])}｜AI三日看多 {int(row['up_votes_3d'])}/{int(row['valid_seeds'])}｜D3 {pred.loc[3,'mean_ret']:+.2f}%")
    caption = (f"📊 Kronos 大盤反轉觀察｜{origin:%Y/%m/%d}\n" + "\n".join(labels)
               + "\n⚠️ 觀察訊號不等於價格已確認反轉，也不是自動交易建議。")
    with IMAGE.open("rb") as image_file:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": (IMAGE.name, image_file, "image/png")},
            timeout=45,
        )
    # Never print the request URL: it contains the secret bot token.
    if response.status_code != 200:
        raise RuntimeError(f"Telegram sendPhoto failed (HTTP {response.status_code}); check bot token/chat ID")
    try:
        ok = response.json().get("ok", False)
    except ValueError:
        ok = False
    if not ok:
        raise RuntimeError("Telegram rejected the report; check bot permissions and chat ID")
    print("[REPORT] Telegram photo sent", flush=True)
    return True


def main():
    if not OUT.exists():
        raise FileNotFoundError("Run Kronos live forecast before this report step")
    origin, summaries = read_inputs()
    make_report(origin, summaries)
    print(f"[REPORT] created {IMAGE} and {REPORT}", flush=True)
    if os.getenv("REPORT_TELEGRAM", "1").strip() == "1":
        send_telegram(origin, summaries)


if __name__ == "__main__":
    main()
