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


def compute_key_levels(ticker: str, trade_date: str) -> str:
    """Return a five-line markdown block of canonical price levels for ``ticker``.

    Levels included: latest close, 50-DMA, 200-DMA, 52-week high/low, and
    20-day high/low. The 20-day range is the canonical short-horizon
    breakout/support proxy for swing setups; the 52-week range and the
    moving averages anchor longer-horizon framing.

    Returns an empty string if the trade_date is malformed, yfinance fails,
    or the response has fewer than 30 rows of data.
    """
    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return ""

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
        logger.warning("compute_key_levels: yfinance fetch failed for %s (%s)", ticker, exc)
        return ""

    if df is None or df.empty or len(df) < 30:
        return ""

    close = df["Close"]
    last_close = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    window_52w = df.tail(252) if len(df) >= 252 else df
    high_52w = float(window_52w["High"].max())
    low_52w = float(window_52w["Low"].min())

    window_20d = df.tail(20)
    high_20d = float(window_20d["High"].max())
    low_20d = float(window_20d["Low"].min())

    return (
        "**Key Price Levels (deterministic, fetched from yfinance — cite these "
        "verbatim if you need a number not surfaced by the upstream debate):**\n"
        f"- Latest close: {_fmt(last_close)}\n"
        f"- 50-DMA: {_fmt(sma50)} | 200-DMA: {_fmt(sma200)}\n"
        f"- 52-week range: {_fmt(low_52w)} – {_fmt(high_52w)}\n"
        f"- 20-day range: {_fmt(low_20d)} – {_fmt(high_20d)} (canonical "
        "breakout/support proxy)"
    )
