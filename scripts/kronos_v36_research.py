"""V3.6 isolated, walk-forward research on frozen V3.5 origin-level forecasts.

No model retraining and no live order generation. Dates are sorted chronologically;
training labels are embargoed until their entire forecast horizon has elapsed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"rank","tw_origin","session","market","horizon","pred_return_pct","realized_market_return_pct"}
THRESHOLDS = (0.5, 1.0)
MIN_TRAIN = 120
WINDOW = 500

def evaluate(frame: pd.DataFrame, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame.duplicated(["rank","session","market","horizon"]).any():
        raise ValueError("Empty or duplicated origin predictions")
    df = frame.copy()
    df["tw_origin"] = pd.to_datetime(df["tw_origin"], errors="raise")
    if df[list(REQUIRED - {"tw_origin","session","market"})].isna().any().any():
        raise ValueError("Null numerical input")
    if not np.isfinite(df[["pred_return_pct","realized_market_return_pct"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite returns")
    results = []
    for (session, market, horizon), group in df.groupby(["session","market","horizon"]):
        g = group.sort_values("tw_origin").reset_index(drop=True)
        if g.tw_origin.duplicated().any():
            raise ValueError(f"Duplicate dates: {session}/{market}/{horizon}")
        # At position i, outcomes for the preceding horizon observations are
        # withheld. This is conservative for non-daily Taiwan origin gaps.
        h = int(horizon)
        for i in range(min_train + h, len(g)):
            train = g.iloc[max(0, i-h-WINDOW):i-h]
            if len(train) < min_train:
                continue
            row = g.iloc[i]
            bias = float((train.realized_market_return_pct - train.pred_return_pct).median())
            pred = float(row.pred_return_pct)
            actual = float(row.realized_market_return_pct)
            # Trend proxy is trailing realized 1-step origin return, not a MA20
            # of historical spot prices. For h>1, do not use h-step returns.
            history = df[(df.session == session) & (df.market == market) &
                         (df.horizon == 1) & (df.tw_origin < row.tw_origin)].sort_values("tw_origin")
            history = history.iloc[:-1] if len(history) else history
            trend = float(history.realized_market_return_pct.tail(20).mean()) if len(history) >= 20 else np.nan
            candidates = {
                "raw": np.sign(pred),
                "calibrated": np.sign(pred + bias),
                "confidence_0p5": np.sign(pred + bias) if abs(pred + bias) >= THRESHOLDS[0] else 0,
                "confidence_1p0": np.sign(pred + bias) if abs(pred + bias) >= THRESHOLDS[1] else 0,
                "trend_filtered": np.sign(pred + bias) if np.isfinite(trend) and np.sign(pred + bias) == np.sign(trend) else 0,
                "always_up": 1,
                "always_down": -1,
            }
            for strategy, direction in candidates.items():
                direction = int(direction)
                results.append({
                    "session":session,"market":market,"horizon":h,"tw_origin":row.tw_origin.date().isoformat(),
                    "strategy":strategy,"signal":direction,"actual_return_pct":actual,
                    "correct":int(np.sign(actual)==direction) if direction else np.nan,
                    "bias_pp":bias,"trend_proxy_pp":trend,
                })
    if not results:
        raise ValueError("Not enough historical origins for walk-forward evaluation")
    return pd.DataFrame(results)

def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session, market, horizon, strategy), g in events.groupby(["session","market","horizon","strategy"]):
        active = g[g.signal != 0]
        rows.append({"session":session,"market":market,"horizon":horizon,"strategy":strategy,
                     "evaluation_origins":len(g),"signals":len(active),
                     "coverage":len(active)/len(g),
                     "direction_accuracy":float(active.correct.mean()) if len(active) else np.nan,
                     "mean_signed_underlying_return_pct":float((active.signal*active.actual_return_pct).mean()) if len(active) else np.nan})
    return pd.DataFrame(rows).sort_values(["session","market","horizon","strategy"])

def run(inputs: list[str], out: str) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for file in inputs:
        source = Path(file)
        stage = source.stem.split("_")[-1]
        if stage not in {"100","500","1000","2000"}:
            raise ValueError(f"Unexpected stage input: {source}")
        frame = pd.read_csv(source)
        if frame["rank"].nunique() != int(stage):
            raise ValueError(f"Incomplete V3.5 stage: {stage}")
        # The smallest stage is deliberately diagnostic-only: insufficient
        # history to estimate a defensible 120-observation calibration.
        if int(stage) <= MIN_TRAIN + 5:
            continue
        events = evaluate(frame)
        events.to_csv(root/f"v36_events_{stage}.csv",index=False)
        summarize(events).to_csv(root/f"v36_summary_{stage}.csv",index=False)
    (root/"README.md").write_text(
        "# V3.6 exploratory calibration research\n\n"
        "All results use frozen V3.5 predictions and known subsequent index returns. "
        "Signals are not executable trade fills or net strategy P&L. "
        "Thresholds are prespecified (0.5/1.0 percentage points); median bias uses "
        "only prior, horizon-embargoed labels (500-origin rolling window). "
        "Trend filter uses a lagged 20-origin mean of 1-step realized returns, "
        "not a market MA20. Missing US_CLOSE dates, overlapping horizons, "
        "pretraining overlap and sample selection limit inference. "
        "Stages 500/1000/2000 are nested, NOT independent out-of-sample sets. "
        "Do not select a winner using these results and claim it is unseen data.\n",
        encoding="utf-8")
    print("V3.6 exploratory research completed; see CSV outputs.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--out",default="outputs_v36")
    a=p.parse_args()
    run(a.inputs,a.out)

if __name__=="__main__":
    main()
