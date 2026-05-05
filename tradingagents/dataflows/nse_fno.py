"""NSE F&O (futures & options) EOD snapshot tool.

NSE's live option-chain endpoints are no longer reliably reachable for
unauthenticated callers, but the EOD bhavcopy CSV archive at
nsearchives.nseindia.com is. That single ZIP contains every contract for
the day, so a download per analyst run is all we need to compute total OI,
Put-Call Ratio, max pain, and the top OI strikes on each side.

Used as a market_analyst signal for swing-horizon Indian setups —
PCR < 0.7 with rising OI is canonical bullish derivative positioning;
PCR > 1.3 the inverse. Build-up at strikes near spot reads as
support/resistance.
"""

from __future__ import annotations

import io
import logging
import threading
import zipfile
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .nse_client import is_indian_ticker, nse_session, strip_nse_suffix

logger = logging.getLogger(__name__)

_BHAV_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

_cache_lock = threading.Lock()
_bhav_cache: dict[str, pd.DataFrame] = {}


def _most_recent_business_days(today: datetime, n: int = 5) -> list[datetime]:
    out: list[datetime] = []
    d = today
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def _fetch_bhav(date: datetime) -> pd.DataFrame | None:
    """Download and parse one EOD F&O bhavcopy. Cached per date."""
    key = date.strftime("%Y%m%d")
    with _cache_lock:
        if key in _bhav_cache:
            return _bhav_cache[key]

    url = _BHAV_URL_TEMPLATE.format(yyyymmdd=key)
    try:
        sess = nse_session()
        r = sess.get(url, headers={"Referer": "https://www.nseindia.com/"}, timeout=20)
        if r.status_code != 200 or len(r.content) < 1000:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f)
    except Exception as e:
        logger.warning("F&O bhavcopy fetch failed for %s: %s", key, e)
        return None

    with _cache_lock:
        _bhav_cache[key] = df
    return df


def _latest_bhav() -> tuple[pd.DataFrame | None, datetime | None]:
    """Return the freshest available bhavcopy + its trade date.

    F&O bhavcopy publishes ~6pm IST. If today's isn't up, walk back day by day.
    """
    for d in _most_recent_business_days(datetime.now(), n=5):
        df = _fetch_bhav(d)
        if df is not None and len(df) > 0:
            return df, d
    return None, None


def _max_pain(strikes: list[float], ce_oi_by_strike: dict[float, int], pe_oi_by_strike: dict[float, int]) -> float | None:
    """Strike that minimises total option-writer payout (≈ max pain)."""
    if not strikes:
        return None
    best = (None, float("inf"))
    for K in strikes:
        # Sum of pain across all strikes if expiry settles at K
        total = 0.0
        for s in strikes:
            ce_oi = ce_oi_by_strike.get(s, 0)
            pe_oi = pe_oi_by_strike.get(s, 0)
            if K > s:
                total += (K - s) * ce_oi
            elif K < s:
                total += (s - K) * pe_oi
        if total < best[1]:
            best = (K, total)
    return best[0]


def get_fno_oi_nse(ticker: str) -> str:
    """EOD F&O snapshot — total OI, PCR, max pain, top-OI strikes.

    Args:
        ticker: yfinance-style ticker (e.g. 'RELIANCE.NS'). Index symbols
            ('NIFTY.NS', 'BANKNIFTY.NS') are also accepted but in practice
            stock tickers are the typical caller.

    Returns:
        Markdown-formatted snapshot. Returns "not applicable" notices for
        non-Indian tickers and for Indian stocks not in the F&O segment.
    """
    if not is_indian_ticker(ticker):
        return (
            f"## F&O OI: not applicable for {ticker}\n\n"
            f"This tool covers NSE F&O segment only (Indian listings)."
        )

    symbol = strip_nse_suffix(ticker)
    df, date = _latest_bhav()
    if df is None:
        return (
            f"## F&O OI for {symbol}\n\n"
            f"NSE F&O bhavcopy unavailable — typically published ~6pm IST. "
            f"Try again later in the trading day."
        )

    # FinInstrmTp: STO = Stock Options, IDO = Index Options, STF = Stock Futures, IDF = Index Futures
    options = df[(df["TckrSymb"] == symbol) & (df["FinInstrmTp"].isin(["STO", "IDO"]))]
    if options.empty:
        return (
            f"## F&O OI for {symbol} ({date.strftime('%d-%b-%Y')} EOD)\n\n"
            f"No F&O contracts found — this name is likely not in the NSE F&O segment."
        )

    # Pick the nearest expiry by date.
    options = options.copy()
    options["XpryDt"] = pd.to_datetime(options["XpryDt"], errors="coerce")
    options = options.dropna(subset=["XpryDt"])
    nearest_expiry = options["XpryDt"].min()
    near = options[options["XpryDt"] == nearest_expiry]

    ce = near[near["OptnTp"] == "CE"][["StrkPric", "OpnIntrst", "ChngInOpnIntrst"]]
    pe = near[near["OptnTp"] == "PE"][["StrkPric", "OpnIntrst", "ChngInOpnIntrst"]]

    total_ce_oi = int(ce["OpnIntrst"].sum())
    total_pe_oi = int(pe["OpnIntrst"].sum())
    pcr = (total_pe_oi / total_ce_oi) if total_ce_oi else None

    ce_by_strike = {float(r.StrkPric): int(r.OpnIntrst) for r in ce.itertuples()}
    pe_by_strike = {float(r.StrkPric): int(r.OpnIntrst) for r in pe.itertuples()}
    all_strikes = sorted(set(ce_by_strike) | set(pe_by_strike))
    mp = _max_pain(all_strikes, ce_by_strike, pe_by_strike)

    # Underlying spot from any future row for the same symbol (closing price)
    futs = df[(df["TckrSymb"] == symbol) & (df["FinInstrmTp"].isin(["STF", "IDF"]))]
    spot: float | None = None
    if not futs.empty and "ClsPric" in futs.columns:
        try:
            spot = float(futs["ClsPric"].iloc[0])
        except (ValueError, TypeError):
            spot = None

    # Top OI strikes — these read as resistance (CE) and support (PE)
    def top_strikes(df_side: pd.DataFrame, n: int = 3) -> list[tuple[float, int, int]]:
        s = df_side.sort_values("OpnIntrst", ascending=False).head(n)
        return [
            (float(r.StrkPric), int(r.OpnIntrst), int(r.ChngInOpnIntrst))
            for r in s.itertuples()
        ]

    top_ce = top_strikes(ce)
    top_pe = top_strikes(pe)

    pcr_label = f"{pcr:.2f}" if pcr is not None else "—"
    pcr_read = ""
    if pcr is not None:
        if pcr < 0.7:
            pcr_read = " — call-heavy positioning (often bullish-contrarian or put-writers in control)"
        elif pcr > 1.3:
            pcr_read = " — put-heavy positioning (often bearish-contrarian or fear elevated)"
        else:
            pcr_read = " — balanced"

    out = [f"## F&O OI snapshot for {symbol} ({date.strftime('%d-%b-%Y')} EOD)\n"]
    out.append(f"Nearest expiry: **{nearest_expiry.date().isoformat()}**")
    if spot is not None:
        out.append(f"Spot (futures close): **{spot:,.2f}**")
    if mp is not None:
        out.append(f"Max pain strike: **{mp:,.2f}**")
    out.append(f"Total Call OI: {total_ce_oi:,}")
    out.append(f"Total Put OI: {total_pe_oi:,}")
    out.append(f"**PCR: {pcr_label}**{pcr_read}\n")

    out.append("### Top 3 Call OI strikes (read as resistance)")
    out.append("| Strike | OI | Δ OI |")
    out.append("|---:|---:|---:|")
    for K, oi, doi in top_ce:
        out.append(f"| {K:,.2f} | {oi:,} | {doi:+,} |")
    out.append("")
    out.append("### Top 3 Put OI strikes (read as support)")
    out.append("| Strike | OI | Δ OI |")
    out.append("|---:|---:|---:|")
    for K, oi, doi in top_pe:
        out.append(f"| {K:,.2f} | {oi:,} | {doi:+,} |")

    return "\n".join(out)
