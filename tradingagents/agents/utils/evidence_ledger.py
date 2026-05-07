"""Structured Evidence Ledger threaded through agent state.

The graph already passes prose reports and small deterministic strings
(``key_levels``, ``market_regime``) between agents, but Research Manager,
Trader, and Portfolio Manager don't see the analyst reports directly.
Numbers can drift across LLM hops — copied wrong, rounded, omitted, or
hallucinated. The ledger gives those downstream nodes a typed surface of
decision-critical facts: a stable, source-attributed shape they can cite
without inventing provenance.

This module defines the typed surface and the deterministic construction
path. Per-agent prompt-injection lives in the consumer modules so each
decision agent renders the ledger when it needs it.

Design notes:

- Pydantic ``BaseModel`` matches the existing schemas in
  :mod:`tradingagents.agents.schemas` so memory-log serialization, copy
  semantics (``model_copy``), and validation share the same shape.
- ``EvidenceFact`` carries a value plus the source label so prompts can
  cite the origin (``yfinance``, ``market_analyst``, ``india_regime``)
  without each downstream prompt re-deriving provenance.
- ``extras`` keeps the schema additive: future facts (Chandelier stop,
  PCR, shareholding) can land via ``merge_facts`` without breaking the
  consumers that read the canonical fields.
- Every helper degrades gracefully: missing data renders to nothing
  rather than raising, so the graph keeps working when yfinance is down.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from tradingagents.agents.utils import market_levels


class EvidenceFact(BaseModel):
    """A single typed fact with a source label.

    ``value`` is intentionally permissive (number or string) so the same
    container holds prices (``200.50``) and qualitative tags
    (``"BULLISH-FAN"``). ``source`` is a short label naming who computed
    the fact (``yfinance``, ``india_regime``, ``market_analyst``,
    etc.); it is surfaced in renderings so the reader can trust the
    provenance.
    """

    value: Optional[Union[float, str]] = None
    source: str
    as_of: Optional[str] = None


class EvidenceLedger(BaseModel):
    """Compact typed surface of decision-critical facts threaded through state.

    Built deterministically at the start of the graph; consumed by
    Research Manager, Trader, and Portfolio Manager as a dependable fact
    surface alongside analyst prose. Canonical fields cover the price
    anchors and regime context that the existing ``key_levels`` /
    ``market_regime`` strings already encode; ``extras`` is the additive
    seam for future facts (Chandelier stop, RS rating, PCR, shareholding)
    without schema churn.
    """

    ticker: str
    trade_date: str

    latest_close: Optional[EvidenceFact] = None
    sma_50: Optional[EvidenceFact] = None
    sma_200: Optional[EvidenceFact] = None
    high_52w: Optional[EvidenceFact] = None
    low_52w: Optional[EvidenceFact] = None
    high_20d: Optional[EvidenceFact] = None
    low_20d: Optional[EvidenceFact] = None

    regime_summary: Optional[str] = None

    extras: dict[str, EvidenceFact] = Field(default_factory=dict)


def build_initial_ledger(ticker: str, trade_date: str) -> EvidenceLedger:
    """Construct a deterministic ledger from the same yfinance fetch
    that powers ``compute_key_levels``.

    On any fetch failure the ledger is returned with ticker + trade_date
    set and all facts as ``None`` — the graph degrades gracefully and the
    renderer emits nothing.
    """
    raw = market_levels._fetch_levels_data(ticker, trade_date)
    if not raw:
        return EvidenceLedger(ticker=ticker, trade_date=trade_date)

    def _fact(key: str) -> Optional[EvidenceFact]:
        v = raw.get(key)
        if v is None:
            return None
        return EvidenceFact(value=v, source="yfinance", as_of=trade_date)

    return EvidenceLedger(
        ticker=ticker,
        trade_date=trade_date,
        latest_close=_fact("latest_close"),
        sma_50=_fact("sma_50"),
        sma_200=_fact("sma_200"),
        high_52w=_fact("high_52w"),
        low_52w=_fact("low_52w"),
        high_20d=_fact("high_20d"),
        low_20d=_fact("low_20d"),
    )


def merge_facts(ledger: EvidenceLedger, facts: dict[str, EvidenceFact]) -> EvidenceLedger:
    """Return a new ledger with ``facts`` merged into ``extras``.

    Non-mutating by design: callers can layer analyst-extracted facts on
    top of the deterministic foundation without rewriting earlier state.
    Existing keys in ``extras`` are overwritten by the supplied dict.
    """
    cloned = ledger.model_copy(deep=True)
    cloned.extras.update(facts)
    return cloned


def _fmt(value: Optional[Union[float, str]]) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_evidence_ledger(ledger: EvidenceLedger) -> str:
    """Render the ledger to a compact markdown block for prompt injection.

    Empty ledgers (no facts, no regime) render to ``""`` so prompts can
    conditionally include them without empty headers leaking through.
    """
    lines: list[str] = []

    if ledger.latest_close is not None:
        lines.append(f"- Latest close: {_fmt(ledger.latest_close.value)}")
    if ledger.sma_50 is not None or ledger.sma_200 is not None:
        sma50 = _fmt(ledger.sma_50.value) if ledger.sma_50 else "n/a"
        sma200 = _fmt(ledger.sma_200.value) if ledger.sma_200 else "n/a"
        lines.append(f"- 50-DMA: {sma50} | 200-DMA: {sma200}")
    if ledger.low_52w is not None or ledger.high_52w is not None:
        lo = _fmt(ledger.low_52w.value) if ledger.low_52w else "n/a"
        hi = _fmt(ledger.high_52w.value) if ledger.high_52w else "n/a"
        lines.append(f"- 52-week range: {lo} – {hi}")
    if ledger.low_20d is not None or ledger.high_20d is not None:
        lo = _fmt(ledger.low_20d.value) if ledger.low_20d else "n/a"
        hi = _fmt(ledger.high_20d.value) if ledger.high_20d else "n/a"
        lines.append(f"- 20-day range: {lo} – {hi}")

    if ledger.regime_summary:
        lines.append(f"- {ledger.regime_summary}")

    for key, fact in sorted(ledger.extras.items()):
        lines.append(f"- {key}: {_fmt(fact.value)} ({fact.source})")

    if not lines:
        return ""
    header = (
        "**Evidence Ledger** (typed facts threaded through state — cite by "
        "name and source):"
    )
    return "\n".join([header, *lines])
