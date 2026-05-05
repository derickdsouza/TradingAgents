"""Shared NSE (National Stock Exchange of India) HTTP client.

NSE's public APIs require a "warm" session — a real-browser User-Agent and
the cookies that nseindia.com sets on first visit. Direct API calls without
warming get 401/403. This module hands out a process-wide cached
requests.Session that has been warmed against the homepage.

The session is recreated on demand if a request fails with 401/403/cookie
issues, so a single stale-cookie blip doesn't break the whole run.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)

_NSE_HOME = "https://www.nseindia.com/"
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Brotli ('br') deliberately omitted — NSE will send Brotli-encoded
    # JSON if offered and `requests` cannot decode it without an optional
    # native package. gzip/deflate are handled out of the box.
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_session_lock = threading.Lock()
_session: requests.Session | None = None


def _build_warmed_session(timeout: float = 10.0) -> requests.Session:
    s = requests.Session()
    s.headers.update(_DEFAULT_HEADERS)
    s.get(_NSE_HOME, timeout=timeout)
    return s


def nse_session(force_refresh: bool = False, timeout: float = 10.0) -> requests.Session:
    """Return a cookie-warmed requests.Session for nseindia.com.

    Args:
        force_refresh: discard any cached session and build a new one.
        timeout: seconds for the homepage warm-up request.
    """
    global _session
    with _session_lock:
        if force_refresh or _session is None:
            _session = _build_warmed_session(timeout=timeout)
        return _session


def nse_get_json(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 15.0,
    retries: int = 1,
) -> Any:
    """GET a JSON endpoint on nseindia.com with a warmed session.

    On 401/403/JSON-decode failure, refresh the session once and retry.

    Args:
        path: path beginning with '/' (e.g. '/api/corporate-announcements').
        params: optional query parameters.
        timeout: per-request timeout in seconds.
        retries: number of refresh-and-retry attempts on auth failures.

    Returns:
        Parsed JSON (dict or list). Raises requests.HTTPError on persistent
        non-recoverable status codes, ValueError on JSON decode failures.
    """
    url = "https://www.nseindia.com" + path
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        sess = nse_session(force_refresh=(attempt > 0))
        try:
            r = sess.get(url, params=params, timeout=timeout, headers={"Referer": _NSE_HOME})
            if r.status_code in (401, 403) and attempt < retries:
                logger.warning("NSE %s -> %s, refreshing session and retrying", path, r.status_code)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            if attempt < retries:
                logger.warning("NSE %s failed (%s), refreshing session and retrying", path, e)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"NSE {path}: exhausted retries without exception")


def strip_nse_suffix(ticker: str) -> str:
    """'TCS.NS' -> 'TCS'; 'RELIANCE.BO' -> 'RELIANCE'; 'AAPL' -> 'AAPL'."""
    t = ticker.upper().strip()
    for suffix in (".NS", ".BO"):
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def is_indian_ticker(ticker: str | None) -> bool:
    if not ticker:
        return False
    t = ticker.upper()
    return t.endswith(".NS") or t.endswith(".BO")
