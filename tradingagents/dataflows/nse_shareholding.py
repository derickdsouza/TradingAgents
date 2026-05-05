"""NSE shareholding-pattern fetcher.

Combines two NSE endpoints to assemble an India-specific shareholding view:

1. /api/corporate-share-holdings-master — quarterly promoter/public history
   (going back ~85 filings). Surfaces promoter % and public % per quarter.
2. /api/corporate-pledgedata — current promoter pledge %, the canonical
   "watch this" Indian fundamentals signal.

A full FII/DII/category breakdown lives inside the XBRL XML referenced by
the master endpoint's `xbrl` field — parsing that is left for a follow-up
because XBRL is heavyweight and the QoQ promoter/pledge view already
covers the most-watched signals.
"""

from __future__ import annotations

import logging
from typing import Any

from .nse_client import is_indian_ticker, nse_get_json, strip_nse_suffix

logger = logging.getLogger(__name__)


def _coerce_pct(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except (ValueError, TypeError):
        return None


def _row(period: str, promoter: float | None, public: float | None, pledge: float | None) -> str:
    def fmt(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "—"
    return f"| {period} | {fmt(promoter)} | {fmt(public)} | {fmt(pledge)} |"


def get_shareholding_pattern_nse(ticker: str, quarters: int = 8) -> str:
    """Fetch quarterly shareholding-pattern data for an NSE-listed ticker.

    Args:
        ticker: yfinance-style ticker (e.g. 'TCS.NS', 'ZEEL.NS').
        quarters: number of past quarters to include in the trend.

    Returns:
        Markdown table with promoter %, public %, and pledge %.
        Non-Indian tickers receive a "not applicable" notice.
    """
    if not is_indian_ticker(ticker):
        return (
            f"## Shareholding pattern: not applicable for {ticker}\n\n"
            f"This tool covers NSE/BSE-listed stocks (suffix .NS or .BO) only."
        )

    symbol = strip_nse_suffix(ticker)

    # Latest pledge snapshot — single row, current quarter.
    pledge_pct: float | None = None
    pledge_date: str = ""
    try:
        pledge_payload = nse_get_json(
            "/api/corporate-pledgedata",
            params={"index": "equities", "symbol": symbol},
        )
        pledge_data = (pledge_payload.get("data") if isinstance(pledge_payload, dict) else None) or []
        if pledge_data:
            row = pledge_data[0]
            pledge_pct = _coerce_pct(row.get("percSharesPledged"))
            pledge_date = (row.get("shp") or "").strip()
    except Exception as e:
        logger.warning("NSE pledge fetch failed for %s: %s", symbol, e)

    # Quarterly history — promoter & public.
    try:
        history = nse_get_json(
            "/api/corporate-share-holdings-master",
            params={"index": "equities", "symbol": symbol},
        )
    except Exception as e:
        logger.warning("NSE shareholding-master fetch failed for %s: %s", symbol, e)
        return (
            f"## Shareholding pattern for {symbol}\n\n"
            f"NSE fetch failed: {e}. Pledge snapshot also unavailable."
            if pledge_pct is None
            else f"## Shareholding pattern for {symbol}\n\n"
                 f"NSE master endpoint failed ({e}); current pledge: "
                 f"{pledge_pct:.2f}% (as of {pledge_date or 'unknown'})."
        )

    if not isinstance(history, list) or not history:
        return f"## Shareholding pattern for {symbol}\n\nNSE returned no quarterly filings."

    # Newest filings come first; take the requested slice.
    rows: list[str] = []
    for entry in history[:quarters]:
        period = (entry.get("date") or "").strip()
        promoter = _coerce_pct(entry.get("pr_and_prgrp"))
        public = _coerce_pct(entry.get("public_val"))
        # Pledge % only known for the most-recent quarter (from the pledge
        # endpoint, which doesn't have history). Match by period to avoid
        # mis-attributing a stale pledge number to older rows.
        pledge_for_row = pledge_pct if (period.lower() == pledge_date.lower() and pledge_pct is not None) else None
        rows.append(_row(period, promoter, public, pledge_for_row))

    table = (
        "| Quarter | Promoter % | Public % | Pledge % |\n"
        "|---|---:|---:|---:|\n" + "\n".join(rows)
    )

    notes: list[str] = []
    if pledge_pct is not None:
        notes.append(
            f"Latest pledge: **{pledge_pct:.2f}%** of total shares "
            f"({pledge_date or 'date unavailable'})."
        )
        if pledge_pct >= 20:
            notes.append("⚠️ High pledge level — promoter pledge above 20% is a known risk signal.")
        elif pledge_pct >= 10:
            notes.append("Pledge level is elevated (>10%); monitor for further increases.")
    notes.append(
        "FII / DII / category breakdown is inside the XBRL filing referenced by NSE; "
        "not surfaced here yet."
    )

    return (
        f"## Shareholding pattern for {symbol} (last {len(rows)} quarters)\n\n"
        f"{table}\n\n" + "\n".join(notes)
    )
