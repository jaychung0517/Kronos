"""V3.5 staging gate: reject incomplete/noisy forecasts and report direction stats.

Run after each staged horizon (100/500/1000/2000). Fail closed if any
historical timestamp uses not-yet-published OHLC or model output is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = {"TAIEX", "TX", "NASDAQ", "SOX"}
HORIZONS = {1, 3, 5}
SEEDS = {0, 1, 2}


def verify(data, total):
    needed = {"rank", "tw_origin", "session", "market", "model_origin",
              "decision_time_tpe", "input_cutoff_tpe", "horizon", "seed_index",
              "random_seed", "pred_return_pct", "realized_market_return_pct",
              "realized_taiex_return_pct", "path_ohlc_valid"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if data.empty or data.isna(subset=list(needed)).any().any():
        raise ValueError("Empty or null forecast results")
    if not set(data["rank"]) == set(range(total)):
        raise ValueError("Rank coverage does not match 0..target-1")
    if set(data.market) != MARKETS or set(data.horizon) != HORIZONS or set(data.seed_index) != SEEDS:
        raise ValueError("Market/horizon/seed coverage incomplete")
    if not set(data.session).issubset({"TW_CLOSE", "US_CLOSE"}):
        raise ValueError("Unexpected decision session")
    keys = ["rank", "session", "market", "horizon", "seed_index"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicated results across shards")
    for (rank, session), group in data.groupby(["rank", "session"]):
        if len(group) != 4 * 3 * 3:
            raise ValueError(f"Missing records on rank={rank}, session={session}: {len(group)}")
        if group.realized_taiex_return_pct.groupby(group.horizon).nunique().max() != 1:
            raise ValueError(f"Target TAIEX realization differs among markets {rank}/{session}")
    tw = data[data.session.eq("TW_CLOSE")]
    if set(tw["rank"]) != set(range(total)):
        raise ValueError("Incomplete Taiwan close coverage")
    rank_dates = tw.groupby("rank").tw_origin.nunique()
    if (rank_dates != 1).any() or not tw.groupby("tw_origin")["rank"].nunique().eq(1).all():
        raise ValueError("Origins nonunique or rank mapping unstable")
    origin_dates = pd.to_datetime(tw.groupby("rank").tw_origin.first().sort_index())
    if not origin_dates.is_monotonic_decreasing or origin_dates.duplicated().any():
        raise ValueError("TW origins not strictly descending in rank")
    for field in ("pred_return_pct", "realized_market_return_pct", "realized_taiex_return_pct"):
        values = pd.to_numeric(data[field], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {field}")
    if (data.pred_return_pct.abs() > 50).any():
        raise ValueError("Model predicted >50% cumulative return; investigate model calibration")
    cutoff = pd.to_datetime(data.input_cutoff_tpe, utc=True)
    decision = pd.to_datetime(data.decision_time_tpe, utc=True)
    if (cutoff > decision).any():
        raise ValueError("LOOKAHEAD: model source close is after prediction decision time")
    if not (data.model_origin.astype(str) <= data.tw_origin.astype(str)).all():
        # For US_CLOSE, the new US session can be same TW calendar date or
        # one day after TW origin. Its *timestamp*, not date string, is validated above.
        usclose = data.session.eq("US_CLOSE") & data.market.isin(["NASDAQ", "SOX"])
        bad = (data.model_origin.astype(str) > data.tw_origin.astype(str)) & ~usclose
        if bad.any():
            raise ValueError("Future market date in Taiwan close/no US update context")
    if not data.path_ohlc_valid.astype(str).str.lower().isin(["true", "false"]).all():
        raise ValueError("Unrecognized OHLC quality flag")
    valid_rate = float(data.path_ohlc_valid.astype(str).str.lower().eq("true").mean())
    if valid_rate < 0.85:
        raise ValueError(f"Predicted OHLC valid rate {valid_rate:.1%} <85%")
    usclose_n = data.loc[data.session.eq("US_CLOSE"), "rank"].nunique()
    if usclose_n < int(total * 0.55):
        raise ValueError(f"Only {usclose_n} of {total} potential US_CLOSE observations")
    return {"origins": total, "raw_records": len(data), "tw_close_origins": len(tw["rank"].unique()),
            "us_close_origins": usclose_n, "ohlc_valid_rate": round(valid_rate, 5)}


def summarize(data):
    groupcols = ["rank", "tw_origin", "session", "market", "horizon"]
    aggregated = data.groupby(groupcols, as_index=False).agg(
        pred_return_pct=("pred_return_pct", "mean"),
        realized_market_return_pct=("realized_market_return_pct", "first"),
        realized_taiex_return_pct=("realized_taiex_return_pct", "first"),
        seed_std_pp=("pred_return_pct", "std"),
        seed_agreement=("pred_return_pct", lambda a: abs(np.sign(a).sum()) / len(a)),
        ohlc_valid_rate=("path_ohlc_valid", "mean"),
    )
    aggregated["pred_direction"] = np.sign(aggregated.pred_return_pct)
    aggregated["actual_direction"] = np.sign(aggregated.realized_market_return_pct)
    aggregated["direction_correct"] = (aggregated.pred_direction == aggregated.actual_direction)
    market_rows = []
    for (market, session, h), g in aggregated.groupby(["market", "session", "horizon"]):
        market_rows.append({
            "market": market, "session": session, "horizon": int(h),
            "n": len(g), "direction_accuracy": float(g.direction_correct.mean()),
            "always_up_baseline": float(g.actual_direction.gt(0).mean()),
            "always_down_baseline": float(g.actual_direction.lt(0).mean()),
            "avg_seed_agreement": float(g.seed_agreement.mean()),
            "pred_up_fraction": float(g.pred_direction.gt(0).mean()),
            "mean_return_mae_pp": float((g.pred_return_pct - g.realized_market_return_pct).abs().mean())
        })
    market_summary = pd.DataFrame(market_rows).sort_values(["session", "market", "horizon"])
    consensus_rows = []
    for (rank, session, h), g in aggregated.groupby(["rank", "session", "horizon"]):
        if len(g) != 4:
            raise ValueError(f"No four-market consensus on {rank}/{session}/{h}")
        signs = g.pred_direction.to_numpy()
        consensus = int(signs[0]) if np.all(signs == signs[0]) and signs[0] != 0 else 0
        taiex_actual = float(g.realized_taiex_return_pct.iloc[0])
        consensus_rows.append({
            "rank": rank, "tw_origin": g.tw_origin.iloc[0], "session": session,
            "horizon": int(h), "consensus": consensus,
            "unanimous": bool(consensus != 0),
            "taiwan_actual_direction": int(np.sign(taiex_actual)),
            "taiwan_actual_return_pct": taiex_actual,
            "correct_if_unanimous": bool(consensus == np.sign(taiex_actual)) if consensus else None,
        })
    consensus = pd.DataFrame(consensus_rows)
    cons_rows = []
    for (session, h), g in consensus.groupby(["session", "horizon"]):
        relevant = g[g.unanimous]
        cons_rows.append({
            "session": session, "horizon": int(h),
            "observation_origins": len(g),
            "four_market_unanimous_count": len(relevant),
            "unanimous_coverage": float(len(relevant)/len(g)),
            "unanimous_correct_pct": float(relevant.correct_if_unanimous.mean()) if len(relevant) else None
        })
    return aggregated, market_summary, consensus, pd.DataFrame(cons_rows)


def run(folder, target, out):
    folder = Path(folder)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob("forecast_shard_*.csv"))
    if not files:
        raise FileNotFoundError("No V3.5 shard files found")
    frames = [pd.read_csv(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    # Stage artifacts may contain additional future ranks; scope strictly to target.
    data = data[data["rank"] < target].copy()
    status = verify(data, target)
    origin_rows, market_summary, consensus, consensus_summary = summarize(data)
    origin_rows.to_csv(out / f"origins_{target}.csv", index=False, encoding="utf-8-sig")
    market_summary.to_csv(out / f"markets_{target}.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(out / f"consensus_events_{target}.csv", index=False, encoding="utf-8-sig")
    consensus_summary.to_csv(out / f"consensus_summary_{target}.csv", index=False, encoding="utf-8-sig")
    (out / f"stage_{target}.json").write_text(
        json.dumps({"success": True, "stage": target, **status}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    md = [
        f"# Kronos V3.5 {target} historical origins (four markets)",
        "",
        f"Validation: PASS. Taiwan-close observations: {status['tw_close_origins']}; "
        f"US-close available: {status['us_close_origins']}; "
        f"predicted OHLC valid: {status['ohlc_valid_rate']:.1%}.",
        "",
        "Prediction directions are from mean of three seeds, not calibrated probabilities.",
        "No parameters are optimized on this evaluation interval.",
        "Overlapping three/five-day horizons are NOT independent samples.",
        "Kronos pretraining may overlap historical evaluation data.",
        "",
        "## Four-market unanimous direction compared with next Taiwan returns",
        "",
        "| Decision | Horizon | Origins | 4/4 observations | Coverage | Accuracy if 4/4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in consensus_summary.itertuples():
        accuracy = f"{row.unanimous_correct_pct:.1%}" if pd.notna(row.unanimous_correct_pct) else "n/a"
        md.append(f"| {row.session} | {row.horizon} | {row.observation_origins} | "
                  f"{row.four_market_unanimous_count} | {row.unanimous_coverage:.1%} | {accuracy} |")
    (out / f"stage_{target}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[V35 GATE] PASSED target={target}, raw_records={len(data)}, US_CLOSE={status['us_close_origins']}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="outputs_v35/shards")
    parser.add_argument("--target", type=int, choices=[100, 500, 1000, 2000], required=True)
    parser.add_argument("--out", default="outputs_v35/stage_reports")
    args = parser.parse_args()
    run(args.shards, args.target, args.out)


if __name__ == "__main__":
    main()
