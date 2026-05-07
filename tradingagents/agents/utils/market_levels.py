"""Compute a tiny canonical set of price anchors for downstream prompts.

Bull/Bear are the only graph nodes that see the full analyst reports;
the Research Manager, Trader, and Portfolio Manager downstream see only
the bull/bear debate transcript and each other's outputs. To keep those
prompts from drifting on numbers, this helper extracts a small
deterministic set of price levels directly from yfinance — bypassing the
LLM entirely — so the same anchors are available everywhere a number
might be cited.

Failure-mode: returns an empty string on any error. Prompts must work
without this block; it is supplemental, not load-bearing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

from tradingagents.dataflows.stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def _fmt(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _fetch_levels_data(ticker: str, trade_date: str) -> Optional[dict]:
    """Fetch raw price-anchor data from yfinance.

    Shared by ``compute_key_levels`` (returns markdown) and
    ``build_initial_ledger`` (returns the typed Evidence Ledger). Splitting
    the fetch from the rendering keeps both consumers on a single yfinance
    call and lets the ledger and the legacy markdown block stay in sync.

    Returns ``None`` on any failure (malformed date, yfinance error, fewer
    than 30 rows of history) so callers can degrade gracefully.
    """
    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None

    start_dt = end_dt - timedelta(days=400)
    try:
        tk = yf.Ticker(ticker.upper())
        df = yf_retry(
            lambda: tk.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        )
    except Exception as exc:
        logger.warning("_fetch_levels_data: yfinance fetch failed for %s (%s)", ticker, exc)
        return None

    if df is None or df.empty or len(df) < 30:
        return None

    close = df["Close"]
    window_52w = df.tail(252) if len(df) >= 252 else df
    window_20d = df.tail(20)

    return {
        "latest_close": float(close.iloc[-1]),
        "sma_50": float(close.tail(50).mean()) if len(close) >= 50 else None,
        "sma_200": float(close.tail(200).mean()) if len(close) >= 200 else None,
        "high_52w": float(window_52w["High"].max()),
        "low_52w": float(window_52w["Low"].min()),
        "high_20d": float(window_20d["High"].max()),
        "low_20d": float(window_20d["Low"].min()),
    }


def compute_key_levels(ticker: str, trade_date: str) -> str:
    """Return a five-line markdown block of canonical price levels for ``ticker``.

    Levels included: latest close, 50-DMA, 200-DMA, 52-week high/low, and
    20-day high/low. The 20-day range is the canonical short-horizon
    breakout/support proxy for swing setups; the 52-week range and the
    moving averages anchor longer-horizon framing.

    Returns an empty string if the trade_date is malformed, yfinance fails,
    or the response has fewer than 30 rows of data.
    """
    raw = _fetch_levels_data(ticker, trade_date)
    if not raw:
        return ""

    return (
        "**Key Price Levels (deterministic, fetched from yfinance — cite these "
        "verbatim if you need a number not surfaced by the upstream debate):**\n"
        f"- Latest close: {_fmt(raw['latest_close'])}\n"
        f"- 50-DMA: {_fmt(raw['sma_50'])} | 200-DMA: {_fmt(raw['sma_200'])}\n"
        f"- 52-week range: {_fmt(raw['low_52w'])} – {_fmt(raw['high_52w'])}\n"
        f"- 20-day range: {_fmt(raw['low_20d'])} – {_fmt(raw['high_20d'])} (canonical "
        "breakout/support proxy)"
    )
