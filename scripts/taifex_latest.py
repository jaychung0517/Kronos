
"""TAIFEX official TX daily-session fallback.

Only supplements a completed trading day.
Night trading is attributed to the following
regular trading day, including holiday/weekend
adjustments.

No brokerage credentials required.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup


TAIFEX_URL = (
    "https://www.taifex.com.tw"
    "/cht/3/futDailyMarketReport"
)


def _number(value):
    text = (
        str(value)
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )

    if not re.fullmatch(
        r"\d+(?:\.\d+)?", text
    ):
        raise ValueError(
            f"Invalid numeric field: {text!r}"
        )

    return float(text)


def parse_tx_session_html(
    html,
    expected_date,
    night
):
    """Parse one official TX session."""

    soup = BeautifulSoup(
        html, "html.parser"
    )

    text = " ".join(
        soup.stripped_strings
    )

    match = re.search(
        r"日期\s*[：:]\s*"
        r"(\d{4}/\d{2}/\d{2})",
        text
    )

    expected = expected_date.strftime(
        "%Y/%m/%d"
    )

    received = (
        match.group(1)
        if match else "unavailable"
    )

    if received != expected:
        raise ValueError(
            "TAIFEX date mismatch: "
            f"expected={expected}, "
            f"received={received}"
        )

    compact = re.sub(
        r"\s+", "", text
    )

    if night:
        marker = (
            "15:00~次日05:00"
            "盤後交易時段行情表"
        )

        if marker not in compact:
            raise ValueError(
                "Wrong TAIFEX night session"
            )

    else:
        if not re.search(
            r"08:45~13:(?:45|30)"
            r"一般交易時段行情表",
            compact
        ):
            raise ValueError(
                "Wrong TAIFEX regular session"
            )

    records = {}

    for table in soup.select(
        "table.table_f"
    ):
        for tr in table.find_all("tr"):
            cells = [
                " ".join(
                    td.stripped_strings
                )
                for td in tr.find_all(
                    "td",
                    recursive=False
                )
            ]

            minimum = (
                15 if night else 17
            )

            if (
                len(cells) < minimum
                or cells[0].strip() != "TX"
            ):
                continue

            contract = cells[1].strip()

            if not re.fullmatch(
                r"20\d{4}",
                contract
            ):
                continue

            month = int(
                contract[-2:]
            )

            if not 1 <= month <= 12:
                continue

            try:
                op, hi, lo, cl = [
                    _number(x)
                    for x in cells[2:6]
                ]

                # 日盤採個別日盤量，
                # 不使用日夜盤合計量。
                volume_index = (
                    8 if night else 9
                )

                vol = _number(
                    cells[volume_index]
                )

            except ValueError:
                continue

            if (
                min(
                    op, hi, lo, cl, vol
                ) <= 0
                or hi < max(
                    op, lo, cl
                )
                or lo > min(
                    op, hi, cl
                )
            ):
                continue

            if contract in records:
                raise ValueError(
                    "Duplicated TAIFEX "
                    f"contract: {contract}"
                )

            records[contract] = {
                "open": op,
                "max": hi,
                "min": lo,
                "close": cl,
                "volume": vol
            }

    if not records:
        raise ValueError(
            "TAIFEX no valid TX OHLCV"
        )

    return records


def fetch_tx_session(
    expected_date,
    night,
    http=None
):
    """Request an exact official session."""

    http = http or requests

    day = expected_date.strftime(
        "%Y/%m/%d"
    )

    market_code = (
        "1" if night else "0"
    )

    payload = {
        "queryType": "2",
        "queryDate": day,
        "marketCode": market_code,
        "MarketCode": market_code,
        "commodity_id": "TX",
        "commodity_idt": "TX",
        "commodity_id2": "",
        "commodity_id2t": "",
        "dateaddcnt": ""
    }

    response = http.post(
        TAIFEX_URL,
        data=payload,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
        timeout=40
    )

    response.raise_for_status()

    if (
        not response.encoding
        or response.encoding.lower()
        == "iso-8859-1"
    ):
        response.encoding = (
            response.apparent_encoding
            or "utf-8"
        )

    return parse_tx_session_html(
        response.text,
        expected_date,
        night
    )


def official_tx_rows(
    expected_date,
    http=None
):
    """Return FinMind-compatible TX rows."""

    day = fetch_tx_session(
        expected_date,
        night=False,
        http=http
    )

    night = fetch_tx_session(
        expected_date,
        night=True,
        http=http
    )

    active_month = (
        expected_date.strftime(
            "%Y%m"
        )
    )

    contracts = sorted(
        contract
        for contract in (
            set(day) & set(night)
        )
        if contract >= active_month
    )

    if not contracts:
        raise ValueError(
            "TAIFEX has no matching "
            "day/night active contract"
        )

    output = []

    for contract in contracts:
        for session, prices in (
            (
                "position",
                day[contract]
            ),
            (
                "after_market",
                night[contract]
            )
        ):
            output.append({
                "date":
                    expected_date.isoformat(),
                "futures_id": "TX",
                "contract_date": contract,
                "trading_session": session,
                **prices
            })

    return pd.DataFrame(output)


def merge_official_tx_latest(
    raw,
    expected_date,
    http=None
):
    """
    Supplement only latest official sessions.
    Preserve all older FinMind history.
    """

    official = official_tx_rows(
        expected_date,
        http=http
    )

    keys = [
        "date",
        "futures_id",
        "contract_date",
        "trading_session"
    ]

    missing = (
        set(keys) - set(raw.columns)
    )

    if missing:
        raise ValueError(
            f"FinMind missing keys: {missing}"
        )

    source = raw.copy()

    for frame in (
        source,
        official
    ):
        frame["date"] = (
            pd.to_datetime(
                frame["date"],
                errors="raise"
            ).dt.strftime(
                "%Y-%m-%d"
            )
        )

        for key in (
            "futures_id",
            "contract_date",
            "trading_session"
        ):
            frame[key] = (
                frame[key]
                .astype(str)
                .str.strip()
            )

    merged = pd.concat(
        [source, official],
        ignore_index=True
    )

    merged = (
        merged.drop_duplicates(
            subset=keys,
            keep="last"
        )
        .reset_index(drop=True)
    )

    if merged.duplicated(
        keys
    ).any():
        raise ValueError(
            "Duplicate rows after merge"
        )

    return merged, official
