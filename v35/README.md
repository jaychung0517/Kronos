# Kronos V3.5 — Isolated Research Workspace

Purpose: audit the longest reliable historical data coverage for TAIEX, TX (regular + night sessions), Nasdaq Composite and Philadelphia Semiconductor Index, then run independent reproducible four-market backtests.

**Do not change production workflows or scripts.** V3.5 code belongs under `scripts/kronos_v35_*`; workflow is `.github/workflows/kronos_v35_backtest.yml`; tests use `tests/test_kronos_v35.py`.

First milestone: data audit only, beginning 2017-05-16 (first TX trading date that may include post-market sessions). No orders, no Telegram, no cloud credentials apart from an optional FINMIND_TOKEN. Actual earliest supported date must be measured, not assumed.
