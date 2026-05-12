"""Tests for ``salvage_into_schema`` — the free-text fallback parser.

Bead TradingAgents-9av: when the structured-output retry loop exhausts, the
raw free-text response was being returned uninterpreted. Renderers downstream
greped key:value pairs out of it field-by-field with silent fallback behavior
that masked structural failures. The salvage parser tries (in order) raw JSON,
fenced ```json blocks, and key:value extraction keyed off the schema's field
names, returning a populated Pydantic instance only when N>=3 fields could be
recovered. Below that threshold the caller emits a ``[SCHEMA_BIND_FAILED]``
sentinel so the failure surfaces loudly instead of silently corrupting prose.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    render_trader_proposal,
)
from tradingagents.agents.utils.freetext_salvage import salvage_into_schema
from tradingagents.agents.utils.structured import invoke_structured_or_freetext


# Real Run 2 trader output — captured verbatim from
# reports/SOUTHBANK.NS/20260512_150227/3_trading/trader.md. This is the
# qwen3.6 free-text response that motivated the bead: silent 50-DMA
# substitution downstream because Entry Price wasn't recognized in prose form.
RUN2_TRADER_BLOB = """\
action: BUY
entry_price: 38.72 INR
entry_basis: 50-DMA/200-DMA cluster support
stop_initial: 36.47 INR
stop_initial_basis: thesis-break (fundamental, fixed)
stop_trailing: 39.85 INR
stop_trailing_basis: AVWAP-52wL / Chandelier Exit (ATR, dynamic)
sizing: 30% initial at anchor, scale +30% on pullback (60% total)
time_horizon: 24 months
rationale: The execution targets the documented 0.92x PBV discount and reserve quality."""


# ---------------------------------------------------------------------------
# Path 1: raw JSON
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPath1RawJson:
    def test_raw_json_payload_returns_populated_instance(self):
        """A response that IS the JSON object should parse via ``json.loads`` and
        validate cleanly. The N>=3 guardrail is comfortably satisfied here.
        """
        payload = {
            "action": "Buy",
            "reasoning": "Trend intact and breakout confirmed.",
            "entry_price": 100.0,
            "entry_basis": "50-DMA",
        }
        result = salvage_into_schema(json.dumps(payload), TraderProposal)
        assert isinstance(result, TraderProposal)
        assert result.action == TraderAction.BUY
        assert result.entry_price == 100.0
        assert result.entry_basis == "50-DMA"


# ---------------------------------------------------------------------------
# Path 2: fenced ```json``` block embedded in prose
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPath2FencedJson:
    def test_json_inside_fenced_block_is_extracted(self):
        """Some models prepend prose, then emit ```json {...} ```. We must find
        the first fenced block, parse it, and ignore surrounding chatter.
        """
        text = (
            "Here is my decision:\n\n"
            "```json\n"
            '{"action": "Sell", "reasoning": "Guidance cut.", '
            '"entry_price": 50.0, "entry_basis": "prior swing high"}\n'
            "```\n\n"
            "Let me know if you want me to revise."
        )
        result = salvage_into_schema(text, TraderProposal)
        assert isinstance(result, TraderProposal)
        assert result.action == TraderAction.SELL
        assert result.entry_price == 50.0


# ---------------------------------------------------------------------------
# Path 3: key:value extraction off schema field names (the main motivator)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPath3KeyValueExtraction:
    def test_run2_trader_blob_salvages_into_trader_proposal(self):
        """The actual Run 2 trader response — pure key:value lines — must
        produce a populated ``TraderProposal`` with action, entry_price, and
        at least one stop. This is the bead's primary motivator: this exact
        blob was returning raw and triggering the silent 50-DMA fallback
        substitution downstream.
        """
        result = salvage_into_schema(RUN2_TRADER_BLOB, TraderProposal)
        assert isinstance(result, TraderProposal), (
            f"salvage missed on Run 2 blob; returned {result!r}"
        )
        assert result.action == TraderAction.BUY
        assert result.entry_price == pytest.approx(38.72)
        # At least one stop must have salvaged.
        assert result.stop_initial is not None or result.stop_trailing is not None
        # The round-tripped render must be non-trivial — the downstream
        # contract is that render_trader_proposal(salvaged) replaces the raw
        # free text.
        rendered = render_trader_proposal(result)
        assert "**Action**: Buy" in rendered
        assert "38.72" in rendered
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in rendered

    def test_pm_blob_salvages_into_portfolio_decision(self):
        """Same salvage path must also handle ``PortfolioDecision`` — both
        decision schemas need to work, not just Trader.
        """
        blob = (
            "rating: Overweight\n"
            "executive_summary: Initiate a 30% position at 38.75 INR.\n"
            "investment_thesis: South Indian Bank presents a deep-value setup.\n"
            "price_target_horizon: 45.5\n"
            "target_basis: mean_reversion\n"
            "time_horizon: 18 months"
        )
        result = salvage_into_schema(blob, PortfolioDecision)
        assert isinstance(result, PortfolioDecision)
        assert result.rating == PortfolioRating.OVERWEIGHT
        assert result.price_target_horizon == pytest.approx(45.5)
        assert result.target_basis == "mean_reversion"


# ---------------------------------------------------------------------------
# Guardrail: N>=3 field threshold (false-positive prevention)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGuardrailFieldThreshold:
    def test_only_two_fields_falls_below_threshold_and_returns_none(self):
        """Blob mentions only two schema field names — below N>=3. Even if
        Pydantic could validate (Optional fields defaulting), salvage must
        return None to protect against false-positive parses.
        """
        blob = (
            "action: BUY\n"
            "reasoning: Brief note, not enough structure for a real proposal."
        )
        result = salvage_into_schema(blob, TraderProposal)
        assert result is None

    def test_empty_string_returns_none(self):
        assert salvage_into_schema("", TraderProposal) is None

    def test_pure_prose_with_no_schema_fields_returns_none(self):
        """Free prose with no key:value structure must not be force-parsed
        into a wrong-but-valid instance.
        """
        prose = (
            "I think the stock looks attractive on the dip. The bank's "
            "balance sheet is strong and management is competent. We could "
            "see a re-rating over the next year."
        )
        assert salvage_into_schema(prose, TraderProposal) is None


# ---------------------------------------------------------------------------
# Integration: salvage wired into invoke_structured_or_freetext
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStructuredFallbackWiring:
    def test_exhausted_retries_with_salvageable_blob_returns_rendered_proposal(self):
        """When structured retries exhaust AND the plain LLM emits a blob the
        salvage parser can recover, the wrapper must return the rendered
        proposal (same shape as the structured success path), NOT the raw
        free text. This proves the salvage is wired in before the return.
        """
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("malformed JSON")

        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content=RUN2_TRADER_BLOB)

        result = invoke_structured_or_freetext(
            structured,
            plain,
            "Trade SOUTHBANK.NS.",
            render_trader_proposal,
            "TestTrader",
            schema=TraderProposal,
            max_retries=2,
        )

        # Not the raw blob, not the sentinel — the rendered proposal.
        assert not result.startswith("[SCHEMA_BIND_FAILED]"), (
            f"sentinel fired on salvageable blob: {result!r}"
        )
        assert "**Action**: Buy" in result
        assert "38.72" in result
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in result

    def test_exhausted_retries_with_irreparable_prose_returns_sentinel(self):
        """When the plain LLM returns prose the salvage parser cannot
        recover, the wrapper must return the ``[SCHEMA_BIND_FAILED]\\n...``
        sentinel — so the downstream renderer (sibling bead fk5) can
        surface the failure loudly instead of silently substituting
        fallback values.
        """
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("malformed JSON")

        irreparable = (
            "I think we should hold this position for now. The market "
            "is volatile and conviction is low."
        )
        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content=irreparable)

        result = invoke_structured_or_freetext(
            structured,
            plain,
            "Trade SOUTHBANK.NS.",
            render_trader_proposal,
            "TestTrader",
            schema=TraderProposal,
            max_retries=2,
        )

        assert result.startswith("[SCHEMA_BIND_FAILED]\n"), (
            f"sentinel missing on irreparable prose: {result!r}"
        )
        # The original response text must follow the sentinel so a human
        # reader can still see what the model emitted.
        assert irreparable in result
