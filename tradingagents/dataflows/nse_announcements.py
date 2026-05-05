"""NSE corporate announcements fetcher.

Pulls the SEBI-mandated disclosures that companies file with NSE — board
meetings, results, dividends, pledge changes, insider trades (Reg 7(2)),
bulk/block deals — which are the actual catalysts that move Indian stocks
and are entirely missing from the Yahoo news pipeline.

Endpoint: https://www.nseindia.com/api/corporate-announcements
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .nse_client import is_indian_ticker, nse_get_json, strip_nse_suffix

logger = logging.getLogger(__name__)


def _parse_nse_dt(s: str | None) -> datetime | None:
    """NSE timestamps look like '06-May-2026 10:32:14' or ISO. Best-effort parse."""
    if not s:
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _format_announcement(item: dict[str, Any]) -> str:
    subject = (item.get("desc") or item.get("subject") or "").strip()
    body = (item.get("attchmntText") or "").strip()
    when = item.get("exchdisstime") or item.get("an_dt") or item.get("sort_date") or ""
    attachment = (item.get("attchmntFile") or "").strip()

    out = f"### {subject or '(no subject)'} — {when}\n"
    if body:
        # Trim runaway descriptions; NSE returns full PDF text on some filings.
        snippet = body if len(body) <= 600 else body[:600].rstrip() + "…"
        out += f"{snippet}\n"
    if attachment:
        out += f"Attachment: {attachment}\n"
    return out + "\n"


def get_corporate_announcements_nse(
    ticker: str,
    look_back_days: int = 30,
    limit: int = 25,
) -> str:
    """Fetch recent corporate announcements for an NSE-listed ticker.

    Args:
        ticker: yfinance-style ticker (e.g. 'TCS.NS', 'RELIANCE.BO'). The .NS/.BO
            suffix is stripped before querying NSE.
        look_back_days: only return announcements newer than this many days.
        limit: maximum number of announcements to return.

    Returns:
        Markdown-formatted string. For non-Indian tickers returns a single-line
        "not applicable" notice so the LLM doesn't try to interpret an error.
    """
    if not is_indian_ticker(ticker):
        return (
            f"## Corporate announcements: not applicable for {ticker}\n\n"
            f"This tool covers NSE/BSE-listed stocks (suffix .NS or .BO) only."
        )

    symbol = strip_nse_suffix(ticker)
    try:
        payload = nse_get_json(
            "/api/corporate-announcements",
            params={"index": "equities", "symbol": symbol},
        )
    except Exception as e:
        logger.warning("NSE announcements fetch failed for %s: %s", symbol, e)
        return (
            f"## Corporate announcements for {symbol}\n\n"
            f"NSE fetch failed: {e}. "
            f"This is often transient — NSE rate-limits unwarmed sessions and the "
            f"endpoint occasionally returns 401/403. Retry on the next analyst pass."
        )

    items = payload if isinstance(payload, list) else payload.get("data") or []
    if not items:
        return f"## Corporate announcements for {symbol}\n\nNo announcements returned by NSE."

    cutoff = datetime.now() - timedelta(days=look_back_days)
    kept: list[dict[str, Any]] = []
    for item in items:
        when = _parse_nse_dt(item.get("exchdisstime") or item.get("sort_date") or item.get("an_dt"))
        if when is None or when >= cutoff:
            kept.append(item)
        if len(kept) >= limit:
            break

    if not kept:
        return (
            f"## Corporate announcements for {symbol} (last {look_back_days}d)\n\n"
            f"None in window. (NSE returned {len(items)} older items.)"
        )

    body = "".join(_format_announcement(it) for it in kept)
    return (
        f"## Corporate announcements for {symbol} "
        f"(last {look_back_days}d, showing {len(kept)} of {len(items)})\n\n{body}"
    )
