"""Compute Indian market regime (broad / cap-tier / sector) for prompt context.

Indian equities trade against three layers of regime: the broad market
(Nifty 50 / 500), the cap tier the stock sits in (Smallcap / Midcap /
Large), and the sector it belongs to. None of these are surfaced
anywhere in the existing pipeline — Minervini's RS-line check uses
Nifty 500 internally but it's binary and buried inside one indicator.

This module fetches all three layers deterministically from yfinance,
computes simple regime tags (BULLISH / NEUTRAL / BEARISH for trend,
with optional RECOVERING / PULLING-BACK / ROLLING-OVER refinements;
LEADING / IN-LINE / LAGGING for relative strength), and renders a
single markdown block that the market analyst, Research Manager,
Trader, and Portfolio Manager prompts all consume.

Failure-mode: returns "" on any error so prompts are never broken by a
data hiccup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import yfinance as yf

from tradingagents.dataflows.nse_client import is_indian_ticker
from tradingagents.dataflows.stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


# Cap-tier thresholds in INR Crore (1 Cr = 1e7 INR). NSE-style buckets.
_LARGE_CAP_MIN_CR = 20_000
_MID_CAP_MIN_CR = 5_000


def _cap_tier(market_cap_inr: Optional[float]) -> tuple[str, Optional[str], Optional[str]]:
    """Map a raw market-cap (INR) to (label, yfinance_symbol, display_name).

    Returns (label, None, None) when market cap is unavailable or zero.
    Display name is what we put in the rendered output.
    """
    if not market_cap_inr or market_cap_inr <= 0:
        return ("Unknown", None, None)
    cr = market_cap_inr / 1e7
    if cr >= _LARGE_CAP_MIN_CR:
        return (f"Large Cap (mcap INR {cr:,.0f} Cr)", "^CNX100", "Nifty 100")
    if cr >= _MID_CAP_MIN_CR:
        return (f"Mid Cap (mcap INR {cr:,.0f} Cr)", "^NSEMDCP50", "Nifty Midcap 50")
    # The plain Nifty Smallcap 100 / 250 index symbols (^CNXSC, ^CRSSMA,
    # NIFTY_SMLCAP_100.NS) all return zero rows from yfinance — the Indian
    # smallcap indices are not surfaced via yfinance the way Nifty 50 / 500
    # are. The Motilal Oswal Nifty Smallcap 250 ETF (MOSMALL250.NS) tracks
    # the index closely and IS available with full history, so it is the
    # working proxy. The label is updated to be honest about that.
    return (
        f"Small Cap (mcap INR {cr:,.0f} Cr)",
        "MOSMALL250.NS",
        "Nifty Smallcap 250 (via Motilal Oswal ETF)",
    )


# yfinance sector + industry → NSE sectoral index. The sector field is
# coarse (~10 buckets); industry disambiguates within Healthcare, Energy,
# Consumer Cyclical, and Financial Services.
#
# Symbol selection note: yfinance only reliably exposes history for the
# legacy ^CNX* / ^NSE* tickers. The newer NIFTY_*.NS index symbols
# (HEALTHCARE, OIL_AND_GAS, CONSR_DURBL, SMLCAP_*) typically return only
# the latest bar and break the moving-average / RS computations. So we
# stick to the legacy tickers and fall back to broader buckets when no
# narrow index has reliable history (e.g. Healthcare → Pharma, Oil/Gas →
# Energy, Consumer Durables → fall through to a no-match note).
_SECTOR_RULES: list[tuple[str, str, str, str]] = [
    # (sector_match, industry_substring_or_empty, yf_symbol, display_name)
    ("Technology", "", "^CNXIT", "Nifty IT"),
    ("Communication Services", "", "^CNXMEDIA", "Nifty Media"),
    ("Healthcare", "", "^CNXPHARMA", "Nifty Pharma"),
    ("Financial Services", "PSU", "^CNXPSUBANK", "Nifty PSU Bank"),
    ("Financial Services", "", "^NSEBANK", "Nifty Bank"),
    ("Consumer Defensive", "", "^CNXFMCG", "Nifty FMCG"),
    ("Consumer Cyclical", "Auto", "^CNXAUTO", "Nifty Auto"),
    ("Basic Materials", "", "^CNXMETAL", "Nifty Metal"),
    ("Energy", "", "^CNXENERGY", "Nifty Energy"),
    ("Industrials", "", "^CNXINFRA", "Nifty Infrastructure"),
    ("Real Estate", "", "^CNXREALTY", "Nifty Realty"),
    ("Utilities", "", "^CNXENERGY", "Nifty Energy"),
]


def _resolve_sector_index(sector: str, industry: str) -> tuple[Optional[str], Optional[str]]:
    """Pick (yf_symbol, display_name) for a yfinance sector/industry pair."""
    for sec, ind_sub, sym, name in _SECTOR_RULES:
        if sec.lower() != sector.lower():
            continue
        if ind_sub and ind_sub.lower() not in industry.lower():
            continue
        return (sym, name)
    return (None, None)


def _fetch_close_series(symbol: str, end_date: datetime) -> Optional[list[float]]:
    """Fetch ~1y of daily closes for symbol. Returns None on failure."""
    start = end_date - timedelta(days=400)
    try:
        df = yf_retry(
            lambda: yf.Ticker(symbol).history(
                start=start.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        )
    except Exception as exc:
        logger.warning("india_regime: fetch failed for %s (%s)", symbol, exc)
        return None
    if df is None or df.empty or len(df) < 30:
        return None
    return [float(x) for x in df["Close"].tolist()]


def _trend_tag(closes: list[float]) -> str:
    """Trend regime tag based on price vs 50/200-DMA.

    Main labels: BULLISH / BEARISH / NEUTRAL. The in-between states get a
    parenthetical refinement so the reader doesn't have to interpret a
    bare NEUTRAL:

    - BULLISH: full stack (last > 50-DMA > 200-DMA, last > 200-DMA) —
      confirmed uptrend.
    - BEARISH: last below both averages AND 50-DMA below 200-DMA —
      confirmed downtrend.
    - BEARISH (ROLLING-OVER): last below both averages but 50-DMA still
      above 200-DMA — fresh breakdown of a prior uptrend.
    - NEUTRAL (RECOVERING): last reclaimed 50-DMA but is still below
      200-DMA, or just reclaimed both averages while the 50/200 stack
      has not yet flipped — a bounce off the lows.
    - NEUTRAL (PULLING-BACK): last above 200-DMA but below 50-DMA —
      short-term pullback inside a longer-term uptrend.
    - NEUTRAL: insufficient data or any other edge case.
    """
    last = closes[-1]
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
    sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
    if sma50 is None or sma200 is None:
        return "NEUTRAL"
    above50 = last > sma50
    above200 = last > sma200
    stack_up = sma50 > sma200
    if above50 and above200 and stack_up:
        return "BULLISH"
    if not above50 and not above200:
        return "BEARISH (ROLLING-OVER)" if stack_up else "BEARISH"
    if above50 and not above200:
        return "NEUTRAL (RECOVERING)"
    if above200 and not above50:
        return "NEUTRAL (PULLING-BACK)"
    if above50 and above200 and not stack_up:
        return "NEUTRAL (RECOVERING)"
    return "NEUTRAL"


def _pct_30d(closes: list[float]) -> Optional[float]:
    """30-trading-day percentage return."""
    if len(closes) < 31:
        return None
    return (closes[-1] / closes[-31] - 1.0) * 100


def _rs_tag(stock_30d: Optional[float], index_30d: Optional[float]) -> str:
    """LEADING / IN-LINE / LAGGING — stock 30d return vs index 30d return."""
    if stock_30d is None or index_30d is None:
        return "n/a"
    diff = stock_30d - index_30d
    if diff >= 2.0:
        return "LEADING"
    if diff <= -2.0:
        return "LAGGING"
    return "IN-LINE"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}%"


def _fmt_level(x: float) -> str:
    return f"{x:,.2f}"


@lru_cache(maxsize=64)
def compute_market_regime(ticker: str, trade_date: str) -> str:
    """Return a markdown regime block for an Indian ticker, "" otherwise.

    Cached by (ticker, trade_date) so the market_analyst's tool-call loop
    can call it on every iteration without refetching yfinance data.
    """
    if not is_indian_ticker(ticker):
        return ""

    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return ""

    stock_closes = _fetch_close_series(ticker, end_dt)
    if stock_closes is None:
        return ""
    stock_30d = _pct_30d(stock_closes)

    try:
        info = yf.Ticker(ticker.upper()).info
    except Exception as exc:
        logger.warning("india_regime: info fetch failed for %s (%s)", ticker, exc)
        info = {}
    market_cap = info.get("marketCap")
    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip()

    lines = ["**Indian Market Regime (deterministic, fetched from yfinance):**"]

    # Broad: Nifty 50 + Nifty 500. Broad-index lines drop silently if
    # yfinance returns nothing — the section below is only useful if at
    # least one broad index resolved.
    for sym, name in (("^NSEI", "Nifty 50"), ("^CRSLDX", "Nifty 500")):
        closes = _fetch_close_series(sym, end_dt)
        if closes is None:
            continue
        lines.append(
            f"- Broad ({name}): {_fmt_level(closes[-1])} | "
            f"30d {_fmt_pct(_pct_30d(closes))} | trend: {_trend_tag(closes)}"
        )

    # Cap tier — only emit a line if the cap-tier index resolves AND its
    # series fetches. Missing data is hidden, not surfaced as "fetch
    # failed" placeholder noise.
    cap_label, cap_sym, cap_name = _cap_tier(market_cap)
    if cap_sym is not None:
        cap_closes = _fetch_close_series(cap_sym, end_dt)
        if cap_closes is not None:
            cap_30d = _pct_30d(cap_closes)
            lines.append(
                f"- Cap tier — {cap_label} → {cap_name}: "
                f"{_fmt_level(cap_closes[-1])} | 30d {_fmt_pct(cap_30d)} | "
                f"stock RS: {_rs_tag(stock_30d, cap_30d)} "
                f"(stock {_fmt_pct(stock_30d)} vs index {_fmt_pct(cap_30d)})"
            )

    # Sector — same rule: only emit a line if both the sector mapping
    # resolves to an NSE symbol AND the series fetches.
    if sector:
        sec_sym, sec_name = _resolve_sector_index(sector, industry)
        sector_label = f"{sector}{' / ' + industry if industry else ''}"
        if sec_sym is not None:
            sec_closes = _fetch_close_series(sec_sym, end_dt)
            if sec_closes is not None:
                sec_30d = _pct_30d(sec_closes)
                lines.append(
                    f"- Sector — {sector_label} → {sec_name}: "
                    f"{_fmt_level(sec_closes[-1])} | 30d {_fmt_pct(sec_30d)} | "
                    f"stock RS: {_rs_tag(stock_30d, sec_30d)} "
                    f"(stock {_fmt_pct(stock_30d)} vs index {_fmt_pct(sec_30d)})"
                )

    # If only the header survived, suppress the whole block — there's
    # nothing useful to surface.
    if len(lines) == 1:
        return ""

    return "\n".join(lines)
