"""Screener.in fundamentals vendor.

Screener.in publishes the cleanest set of Indian-listed company fundamentals
on the open web: 10+ years of P&L / balance sheet / cash flow, key ratios,
shareholding history, with consolidated and standalone variants exposed
side-by-side. yfinance's Indian fundamentals are often stale or incomplete
for mid-caps; Screener.in is the local-analyst standard.

This vendor scrapes the public company page (no API). Page structure is
stable across years (sections keyed by stable HTML ids), but expect
breakage every few years when Screener.in does a redesign — when that
happens disable the vendor by switching data_vendors.fundamental_data
back to yfinance.

Routes registered for: get_fundamentals, get_balance_sheet, get_cashflow,
get_income_statement.
"""

from __future__ import annotations

import logging
import threading
import urllib.parse
import urllib.request
from typing import Any

from bs4 import BeautifulSoup

from .nse_client import is_indian_ticker, strip_nse_suffix

logger = logging.getLogger(__name__)

_BASE = "https://www.screener.in/company"
_USER_AGENT = "Mozilla/5.0 (compatible; TradingAgents/1.0)"
_TIMEOUT = 15.0

_page_cache_lock = threading.Lock()
_page_cache: dict[tuple[str, bool], BeautifulSoup | None] = {}


def _fetch_page(symbol: str, consolidated: bool) -> BeautifulSoup | None:
    key = (symbol, consolidated)
    with _page_cache_lock:
        if key in _page_cache:
            return _page_cache[key]

    suffix = "/consolidated/" if consolidated else "/"
    url = f"{_BASE}/{urllib.parse.quote(symbol)}{suffix}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            if r.status >= 400:
                soup = None
            else:
                soup = BeautifulSoup(r.read(), "lxml")
    except Exception as e:
        logger.warning("Screener fetch failed (%s, consolidated=%s): %s", symbol, consolidated, e)
        soup = None

    with _page_cache_lock:
        _page_cache[key] = soup
    return soup


def _resolve_page(symbol: str) -> tuple[BeautifulSoup | None, str]:
    """Try consolidated first, fall back to standalone. Return (soup, variant)."""
    soup = _fetch_page(symbol, consolidated=True)
    if soup is not None and soup.find(id="profit-loss"):
        return soup, "consolidated"
    soup = _fetch_page(symbol, consolidated=False)
    if soup is not None and soup.find(id="profit-loss"):
        return soup, "standalone"
    return None, ""


def _parse_section(soup: BeautifulSoup, section_id: str) -> list[list[str]] | None:
    section = soup.find(id=section_id)
    if not section:
        return None
    table = section.find("table")
    if not table:
        return None
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        rows.append([c.get_text(strip=True) for c in cells])
    return rows or None


def _format_table(rows: list[list[str]], title: str, max_cols: int = 11) -> str:
    """Format a parsed Screener.in table as markdown, trimming wide year ranges."""
    if not rows:
        return f"### {title}\n\n_No data._\n"
    # Trim columns to the most recent N years (drop oldest mid-cols, keep label + last N).
    if len(rows[0]) > max_cols:
        keep_label = [r[0] for r in rows]
        keep_recent = [r[-(max_cols - 1):] for r in rows]
        rows = [[label, *recent] for label, recent in zip(keep_label, keep_recent)]

    header = rows[0]
    md = [f"### {title}", ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows[1:]:
        # Pad short rows
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md) + "\n"


def _build_report(ticker: str, sections: list[tuple[str, str]]) -> str:
    """Fetch the page and format the requested sections into a single report."""
    if not is_indian_ticker(ticker):
        return (
            f"## Screener.in: not applicable for {ticker}\n\n"
            f"This vendor covers Indian listings (NSE/BSE) only."
        )

    symbol = strip_nse_suffix(ticker)
    soup, variant = _resolve_page(symbol)
    if soup is None:
        return (
            f"## Screener.in fundamentals for {symbol}\n\n"
            f"Page fetch failed or company not on Screener.in. "
            f"This often means the ticker symbol differs (e.g. NSE listing under a "
            f"different short code) — check screener.in/company/{symbol}/ in a browser."
        )

    out: list[str] = [
        f"## Screener.in fundamentals for {symbol} ({variant})",
        f"_Source: https://www.screener.in/company/{symbol}/{'consolidated/' if variant == 'consolidated' else ''}_",
        "",
    ]
    for section_id, title in sections:
        rows = _parse_section(soup, section_id)
        if rows is None:
            out.append(f"### {title}\n\n_Section not present on page._\n")
        else:
            out.append(_format_table(rows, title))
    return "\n".join(out)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Comprehensive fundamentals: P&L, balance sheet, cash flow, ratios, shareholding."""
    del curr_date
    return _build_report(
        ticker,
        sections=[
            ("profit-loss", "Profit & Loss (annual)"),
            ("balance-sheet", "Balance Sheet"),
            ("cash-flow", "Cash Flow"),
            ("ratios", "Key Ratios"),
            ("shareholding", "Shareholding (quarterly)"),
        ],
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Balance sheet (annual rows from Screener.in)."""
    del freq, curr_date
    return _build_report(ticker, sections=[("balance-sheet", "Balance Sheet")])


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Cash flow statement (annual rows from Screener.in)."""
    del freq, curr_date
    return _build_report(ticker, sections=[("cash-flow", "Cash Flow")])


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Income statement / P&L (annual rows from Screener.in).

    Screener.in also has a quarterly P&L section keyed #quarters; surfacing
    that alongside the annual view since the freq=quarterly case asked for it.
    """
    del freq, curr_date
    return _build_report(
        ticker,
        sections=[
            ("profit-loss", "Profit & Loss (annual)"),
            ("quarters", "Quarterly Results"),
        ],
    )
