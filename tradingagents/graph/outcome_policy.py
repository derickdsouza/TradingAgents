"""Outcome resolution policy: holding window and alpha benchmark per
configured trading horizon and ticker region.

The deferred memory loop measures whether a past decision worked. The two
knobs that control "did it work" are:

  - the holding window (how many trading days to look forward)
  - the benchmark used for alpha (so a +5% raw return in a +10% market
    is correctly classified as -5% alpha)

Hardcoding 5 trading days and SPY only makes sense for a US short-horizon
default. For ``position`` (3-6 months) or ``long-term`` (12+ months)
recommendations, a 5-day measurement is noise. For Indian ``.NS`` / ``.BO``
listings, SPY is also the wrong relative.

This module centralises the policy so the trading graph and the reflector
agree on the same window/benchmark, and the choice is captured in the log
for future agents to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Horizon -> trading-day holding window.
# Picked to roughly match the prompt-side holding-period phrasing in
# ``HORIZONS`` (agents/utils/agent_utils.py): swing = 2-6 weeks ~ 20 td,
# position = 3-6 months ~ 63 td (one quarter), long-term = 12+ months ~
# 252 td (one trading year). Long-term entries that have not yet ripened
# stay pending rather than being resolved with a short-window proxy.
HOLDING_DAYS_BY_HORIZON: dict[str, int] = {
    "swing": 20,
    "position": 63,
    "long-term": 252,
}
DEFAULT_HORIZON = "swing"
DEFAULT_HOLDING_DAYS = HOLDING_DAYS_BY_HORIZON[DEFAULT_HORIZON]


@dataclass(frozen=True)
class OutcomePolicy:
    """Resolved (horizon, holding_days, benchmark) triple for a single decision."""

    horizon: str
    holding_days: int
    benchmark: str


def holding_days_for_horizon(horizon: Optional[str]) -> int:
    """Map a horizon key to its trading-day holding window.

    Falls back to the swing default for unknown / missing values so a
    misconfigured horizon never crashes the resolution loop.
    """
    if not horizon:
        return DEFAULT_HOLDING_DAYS
    return HOLDING_DAYS_BY_HORIZON.get(horizon.lower(), DEFAULT_HOLDING_DAYS)


def benchmark_for_ticker(ticker: Optional[str]) -> str:
    """Pick the alpha benchmark for a ticker based on its exchange suffix.

    Mirrors the regional rules used elsewhere (Minervini RS in
    ``dataflows.y_finance._resolve_benchmark``) so memory log alpha and
    indicator-side relative strength stay consistent.
    """
    s = (ticker or "").upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "^CRSLDX"  # Nifty 500
    if s.endswith(".L"):
        return "^FTSE"
    if s.endswith(".HK"):
        return "^HSI"
    if s.endswith(".T"):
        return "^N225"
    if s.endswith(".TO"):
        return "^GSPTSE"
    if s.endswith(".AX"):
        return "^AXJO"
    return "SPY"


def resolve_outcome_policy(config: Optional[dict], ticker: str) -> OutcomePolicy:
    """Resolve the (horizon, holding_days, benchmark) triple to use for one ticker.

    Reads ``config["trading_horizon"]`` (canonicalised to lowercase) and
    derives the regional benchmark from the ticker suffix. Both lookups
    fall back to safe defaults so the resolution loop never throws.
    """
    raw = (config or {}).get("trading_horizon") or DEFAULT_HORIZON
    horizon = str(raw).lower()
    return OutcomePolicy(
        horizon=horizon if horizon in HOLDING_DAYS_BY_HORIZON else DEFAULT_HORIZON,
        holding_days=holding_days_for_horizon(horizon),
        benchmark=benchmark_for_ticker(ticker),
    )
