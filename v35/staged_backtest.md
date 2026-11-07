# Kronos V3.5-B staged four-market backtest

This research stage is isolated from `tw_close`, `us_close`, Telegram, cron-job.org and all V3.4 production Python/YAML files.

## Automated run

The standalone workflow `.github/workflows/kronos_v35_autostages.yml` validates one fixed 2017+ data snapshot, then proceeds:

- 100 latest eligible Taiwan trading origins: rank 0–99
- 500 cumulative: add rank 100–499 (400 previously uncalculated)
- 1,000 cumulative: add rank 500–999
- 2,000 cumulative: add rank 1000–1999

Each worker processes 100 historical Taiwan origins, using the same Kronos-base B parameters (200-bar context, five paths per seed, three seeds), and writes its own seed-level forecast artifact. Workers never update live market output or send Telegram notifications.

**Gates:** fully present four-market/three-seed/1-3-5-day predictions; no future OHLC at decision time; deterministic origin ranking; valid forecast prices; predicted five-bar OHLC quality >=85%; US_CLOSE signal coverage >=55%. On any failure, the next stage is skipped. Poor historical predictive accuracy is reported honestly, not treated as a software failure or tuned away.

**Two decision times:** TW_CLOSE uses TAIEX and TX through that Taiwan regular close plus the latest US session already closed. US_CLOSE uses the same Taiwan bars plus newly completed US session before the next TW market open (including Saturday morning reports for Friday US close). All source-close instants are checked against Asia/Taipei decision times with NY daylight-saving rules. Futures history uses backward adjustment constructed only with observations known at each prediction origin; realized returns use target-date roll accounting for scoring only.

## Research limitations

- Model pretraining may have included historical observations. This is historical behavior research, not necessarily out-of-sample validation.
- Futures adjusted returns are not executable trading P&L; no slippage, fees or derivatives margin.
- 3- and 5-day horizons overlap; counts do not represent independent events.
- Four highly correlated markets agreeing is **not** automatically a calibrated probability.
- Data audit checks 2017+ coverage; actual realized availability may still include revisions.
- Program has no market order actions and no account credentials aside from optional FINMIND_TOKEN.
- The workflow is intentionally compute-intensive and can consume considerable GitHub Actions minutes; maximum parallel workers is two.

## Artifacts
Snapshot, rank-shard forecast CSVs, cumulative stage reports and an accuracy/4-market-consensus summary at each target size. The first failed gate stops all later phases.
