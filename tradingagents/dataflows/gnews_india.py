"""Google News RSS vendor scoped to Indian financial publishers.

Yahoo's news pool under-indexes Indian financial wires (Moneycontrol, Mint,
Business Standard, ET, BQ Prime). This vendor hits Google News RSS with
site-scoped queries so the pool is wide and the publishers are right —
without per-publisher scraping.

This is a news_data vendor and registers under the same get_news /
get_global_news contracts as the yfinance and alpha_vantage vendors.
Stdlib only; no extra packages.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from .nse_client import is_indian_ticker, strip_nse_suffix

logger = logging.getLogger(__name__)

_INDIAN_PUBLISHERS = (
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "economictimes.indiatimes.com",
    "bqprime.com",
    "thehindubusinessline.com",
    "ndtv.com/business",
)

_INDIA_MACRO_QUERIES = (
    "RBI monetary policy",
    "India CPI WPI inflation",
    "FII DII flows India equities",
    "INR USD Brent crude oil India",
)

_GNEWS_BASE = "https://news.google.com/rss/search"
_REQUEST_TIMEOUT = 10.0
_USER_AGENT = "Mozilla/5.0 (compatible; TradingAgents/1.0)"


def _site_clause() -> str:
    return " OR ".join(f"site:{s}" for s in _INDIAN_PUBLISHERS)


def _gnews_url(query: str) -> str:
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    }
    return f"{_GNEWS_BASE}?{urllib.parse.urlencode(params)}"


def _fetch_rss(url: str) -> list[dict[str, str]]:
    """Fetch and parse a Google News RSS feed. Returns a list of items."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as r:
            raw = r.read()
    except Exception as e:
        logger.warning("Google News RSS fetch failed (%s): %s", url, e)
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        logger.warning("Google News RSS parse failed: %s", e)
        return []

    items: list[dict[str, str]] = []
    for it in root.findall(".//item"):
        items.append(
            {
                "title": (it.findtext("title") or "").strip(),
                "link": (it.findtext("link") or "").strip(),
                "pub_date": (it.findtext("pubDate") or "").strip(),
                "source": (it.findtext("source") or "").strip(),
            }
        )
    return items


def _parse_pub_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        return dt.replace(tzinfo=None) if dt else None
    except (TypeError, ValueError):
        return None


def _filter_and_format(
    items: list[dict[str, str]],
    start: datetime,
    end: datetime,
    limit: int,
    header: str,
) -> str:
    seen_titles: set[str] = set()
    formatted: list[str] = []
    for it in items:
        title = it["title"]
        if not title or title in seen_titles:
            continue
        when = _parse_pub_date(it["pub_date"])
        if when and not (start <= when <= end + timedelta(days=1)):
            continue
        seen_titles.add(title)
        publisher = it["source"] or "Unknown"
        line = f"### {title} (source: {publisher})\n"
        if it["pub_date"]:
            line += f"_{it['pub_date']}_\n"
        if it["link"]:
            line += f"Link: {it['link']}\n"
        formatted.append(line + "\n")
        if len(formatted) >= limit:
            break

    if not formatted:
        return f"{header}\n\nNo articles found."
    return f"{header}\n\n" + "".join(formatted)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Per-ticker news from Google News scoped to Indian publishers.

    Only useful for Indian listings. For non-Indian tickers returns a
    "not applicable" notice — callers should fall back to yfinance.
    """
    if not is_indian_ticker(ticker):
        return (
            f"## Google News (India): not applicable for {ticker}\n\n"
            f"This vendor is scoped to Indian publishers."
        )

    symbol = strip_nse_suffix(ticker)
    query = f'"{symbol}" stock ({_site_clause()})'
    items = _fetch_rss(_gnews_url(query))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    header = f"## {symbol} news (Indian publishers via Google News, {start_date} to {end_date})"
    return _filter_and_format(items, start, end, limit=15, header=header)


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
    ticker: str | None = None,
) -> str:
    """India-macro headlines from Google News scoped to Indian publishers.

    Args:
        curr_date: yyyy-mm-dd.
        look_back_days: filter window.
        limit: max articles.
        ticker: ignored for content selection (queries are always India-macro);
            accepted for vendor-interface parity with the yfinance path.
    """
    del ticker

    end = datetime.strptime(curr_date, "%Y-%m-%d")
    start = end - timedelta(days=look_back_days)

    all_items: list[dict[str, str]] = []
    for q in _INDIA_MACRO_QUERIES:
        full_query = f"{q} ({_site_clause()})"
        all_items.extend(_fetch_rss(_gnews_url(full_query)))
        if len(all_items) >= limit * 4:
            break

    header = f"## India macro news (Indian publishers, {start.date()} to {end.date()})"
    return _filter_and_format(all_items, start, end, limit=limit, header=header)


def get_insider_transactions(ticker: str) -> str:
    """Not implemented for this vendor.

    Indian insider disclosures live on NSE/BSE corporate-announcements.
    """
    return (
        f"## Insider transactions for {ticker}: not implemented in gnews_india\n\n"
        f"Indian insider disclosures (SEBI Reg 7(2)) are filed with NSE/BSE; "
        f"see get_corporate_announcements instead."
    )
