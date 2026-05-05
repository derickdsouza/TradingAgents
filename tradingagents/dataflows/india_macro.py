"""India macro-context snapshot.

Returns a tight snapshot of the macro factors that dominate Indian equity
price action: INR/USD, Brent crude, Nifty 50/Bank, India VIX, and the EOD
institutional flow (FII vs DII cash-market net).

Sources:
- yfinance for INR=X, BZ=F, ^NSEI, ^NSEBANK, ^INDIAVIX
- NSE /api/fiidiiTradeReact for daily FII/DII cash-market flows

Why this tool exists: get_global_news returns headlines, but Indian macro
moves on numbers (INR direction, Brent, FII/DII flow today) more than on
headlines. The agent gets both signals, complementary.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from .nse_client import nse_get_json
from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def _yf_close_change(symbol: str) -> tuple[float | None, float | None, float | None]:
    """Return (last_close, 1d_pct, 5d_pct) for a yfinance symbol.

    Falls back to (None, None, None) on error so a single missing series
    doesn't fail the whole macro snapshot.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = yf_retry(lambda: ticker.history(period="10d"))
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None, None, None

    closes = list(hist["Close"].dropna()) if hasattr(hist, "__contains__") and "Close" in hist else []
    if not closes:
        return None, None, None

    last = float(closes[-1])
    one_d = float((last / closes[-2] - 1) * 100) if len(closes) >= 2 else None
    five_d = float((last / closes[-6] - 1) * 100) if len(closes) >= 6 else None
    return last, one_d, five_d


def _fmt(x: float | None, precision: int = 2, suffix: str = "") -> str:
    if x is None:
        return "—"
    return f"{x:,.{precision}f}{suffix}"


def _fmt_signed(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def _fii_dii_flow() -> dict[str, dict[str, float | str]] | None:
    """Most recent EOD FII/DII cash-market net flow from NSE. Crores INR."""
    try:
        payload = nse_get_json("/api/fiidiiTradeReact")
    except Exception as e:
        logger.warning("NSE FII/DII fetch failed: %s", e)
        return None

    if not isinstance(payload, list):
        return None

    out: dict[str, dict[str, float | str]] = {}
    for entry in payload:
        cat = (entry.get("category") or "").upper()
        if cat not in ("FII", "DII"):
            continue
        try:
            out[cat] = {
                "date": entry.get("date") or "",
                "buy": float(entry.get("buyValue") or 0),
                "sell": float(entry.get("sellValue") or 0),
                "net": float(entry.get("netValue") or 0),
            }
        except (ValueError, TypeError):
            continue
    return out or None


def get_india_macro(curr_date: str | None = None) -> str:
    """Snapshot of India macro-context drivers.

    Args:
        curr_date: accepted for interface symmetry with other tools; the
            snapshot is always live (most-recent EOD).

    Returns:
        Markdown-formatted summary table + EOD institutional flow.
    """
    del curr_date

    inr_spot, inr_1d, _ = _yf_close_change("INR=X")
    brent_spot, brent_1d, brent_5d = _yf_close_change("BZ=F")
    nifty, nifty_1d, nifty_5d = _yf_close_change("^NSEI")
    bank_nifty, bank_nifty_1d, _ = _yf_close_change("^NSEBANK")
    vix, _, _ = _yf_close_change("^INDIAVIX")

    flow = _fii_dii_flow()

    lines: list[str] = ["## India macro snapshot\n"]
    lines.append("| Indicator | Level | 1d | 5d |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| INR/USD | {_fmt(inr_spot, 4)} | {_fmt_signed(inr_1d)} | — |")
    lines.append(f"| Brent crude (USD) | {_fmt(brent_spot, 2)} | {_fmt_signed(brent_1d)} | {_fmt_signed(brent_5d)} |")
    lines.append(f"| Nifty 50 | {_fmt(nifty, 2)} | {_fmt_signed(nifty_1d)} | {_fmt_signed(nifty_5d)} |")
    lines.append(f"| Nifty Bank | {_fmt(bank_nifty, 2)} | {_fmt_signed(bank_nifty_1d)} | — |")
    lines.append(f"| India VIX | {_fmt(vix, 2)} | — | — |")
    lines.append("")

    if flow:
        lines.append("### EOD institutional flow (₹ cr, cash market)")
        date = flow.get("FII", {}).get("date") or flow.get("DII", {}).get("date") or ""
        lines.append(f"_As of {date}_\n")
        lines.append("| Category | Buy | Sell | **Net** |")
        lines.append("|---|---:|---:|---:|")
        for cat in ("FII", "DII"):
            f = flow.get(cat)
            if not f:
                continue
            net = f["net"]
            net_label = f"**+{net:,.0f}**" if net >= 0 else f"**{net:,.0f}**"
            lines.append(f"| {cat} | {f['buy']:,.0f} | {f['sell']:,.0f} | {net_label} |")
    else:
        lines.append("_FII/DII flow unavailable from NSE (may be a non-trading day)._")

    lines.append("")
    lines.append(
        "**Reading note:** sustained FII selling with DII buying is the canonical "
        "Indian institutional tug-of-war; net FII outflow + INR weakening + Brent up "
        "is the classic risk-off setup for India equities. India 10Y G-Sec yield is "
        "not surfaced here yet (no clean yfinance symbol)."
    )

    return "\n".join(lines)
