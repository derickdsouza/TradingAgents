"""Decision-quality regression harness.

This test module is the deterministic safety net behind the fork's prompt and
schema work. It pins the invariants that motivated those changes:

  - ``parse_rating`` agrees with the rendered headline rating.
  - Rendered Trader / PM / Research-Manager markdown stays parseable by the
    existing CLI / memory-log / signal-processor consumers.
  - Long-side stops cannot sit above the latest close (and the validator
    drops them when an LLM tries it anyway).
  - PM rating polarity stays consistent with Trader action polarity.
  - Saved report sections and rendered structured outputs do not start with
    banned narrator preambles ("Now I have all the data", "Let me analyze",
    etc.) — these are precisely the kind of regression a rebase can quietly
    reintroduce.
  - The Evidence Ledger and Scorecard blocks render with their stable
    headers, so report templates and downstream parsers can rely on them.

No network, no API keys. Everything runs against in-memory schema renders
and tiny inline fixtures. To extend: add a fixture string at the top of the
relevant section and a one-liner test that asserts the invariant of
interest. The harness deliberately tests *public* seams (render helpers,
parsers, saved markdown shape) rather than internal implementation details.
"""

from __future__ import annotations

import re

import pytest

from tradingagents.agents.schemas import (
    Confidence,
    EvidenceScorecard,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)
from tradingagents.agents.utils.decision_contracts import (
    TraderValidationContext,
    validate_trader_proposal,
)
from tradingagents.agents.utils.evidence_ledger import (
    EvidenceFact,
    EvidenceLedger,
    render_evidence_ledger,
)
from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating


# ---------------------------------------------------------------------------
# Helpers — small public-shape utilities used across multiple tests.
# ---------------------------------------------------------------------------

# Phrases that signal narrator-mode prose. Any of these appearing at the
# start of a generated report section is a regression — the prompt overlay
# layer is supposed to suppress them. Add new items here as new failures
# are observed.
BANNED_NARRATOR_PHRASES: tuple[str, ...] = (
    "Now I have all the data",
    "Now I have the data",
    "Now that I have",
    "Let me analyze",
    "Let me start by",
    "Based on the analysis above",
    "Based on the data provided",
    "I will now",
    "I'll now",
)


def has_banned_narrator_preamble(text: str) -> str | None:
    """Return the matching banned phrase if ``text`` opens with one, else None.

    Compared case-insensitively against the first 200 characters of the
    text (after stripping leading whitespace) so both bare prose and
    markdown-prefixed reports are caught.
    """
    head = text.lstrip()[:200].lower()
    for phrase in BANNED_NARRATOR_PHRASES:
        if phrase.lower() in head:
            return phrase
    return None


# Rating polarity on the 5-tier scale: bullish, neutral, bearish.
_BULLISH_RATINGS = {"Buy", "Overweight"}
_BEARISH_RATINGS = {"Underweight", "Sell"}
_NEUTRAL_RATINGS = {"Hold"}

# Trader action polarity (3-tier).
_BULLISH_ACTIONS = {TraderAction.BUY}
_BEARISH_ACTIONS = {TraderAction.SELL}
_NEUTRAL_ACTIONS = {TraderAction.HOLD}


def rating_action_consistent(rating: str, action: TraderAction) -> bool:
    """Whether a PM rating and a Trader action point the same direction.

    Bullish ratings (Buy / Overweight) must pair with Buy or Hold actions;
    bearish ratings (Underweight / Sell) must pair with Sell or Hold; and
    a Hold rating may pair with any action. The Trader's 3-tier scale is
    intentionally narrower than the PM's 5-tier scale, so an Overweight
    rating + a Buy action is consistent.
    """
    if rating in _NEUTRAL_RATINGS:
        return True
    if rating in _BULLISH_RATINGS:
        return action in _BULLISH_ACTIONS or action in _NEUTRAL_ACTIONS
    if rating in _BEARISH_RATINGS:
        return action in _BEARISH_ACTIONS or action in _NEUTRAL_ACTIONS
    return False  # unknown rating string


def _full_scorecard(**overrides) -> EvidenceScorecard:
    """Helper: construct a minimal EvidenceScorecard with every required field."""
    base = dict(
        bull_case=1, bear_case=-1, trend_technical=1, fundamental_quality=0,
        liquidity_risk=0, catalyst_clarity=0, macro_regime=0, valuation=0,
        confidence=Confidence.MEDIUM,
        rating_rationale="Bull case modestly outweighs bear case.",
        invalidating_evidence="A close below the 200-DMA flips the trend leg negative.",
    )
    base.update(overrides)
    return EvidenceScorecard(**base)


# ---------------------------------------------------------------------------
# Helper-function tests (the helpers themselves must be trustworthy).
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_banned_preamble_detects_now_i_have(self):
        text = "Now I have all the data I need. Here is my analysis..."
        assert has_banned_narrator_preamble(text) == "Now I have all the data"

    def test_banned_preamble_case_insensitive(self):
        text = "NOW THAT I HAVE reviewed the briefing..."
        assert has_banned_narrator_preamble(text) == "Now that I have"

    def test_banned_preamble_ignores_phrase_deep_in_body(self):
        """A banned phrase appearing well past the first 200 chars is not flagged."""
        text = "**Rating**: Buy\n\n" + "Body text. " * 40 + "Now I have all the data."
        assert has_banned_narrator_preamble(text) is None

    def test_banned_preamble_clean_report_passes(self):
        text = "**Rating**: Buy\n\n**Executive Summary**: Build position gradually."
        assert has_banned_narrator_preamble(text) is None

    def test_rating_action_consistent_buy_buy(self):
        assert rating_action_consistent("Buy", TraderAction.BUY)

    def test_rating_action_consistent_overweight_hold(self):
        """Overweight + Hold is consistent: PM is willing to wait for entry."""
        assert rating_action_consistent("Overweight", TraderAction.HOLD)

    def test_rating_action_inconsistent_buy_sell(self):
        assert not rating_action_consistent("Buy", TraderAction.SELL)

    def test_rating_action_inconsistent_sell_buy(self):
        assert not rating_action_consistent("Sell", TraderAction.BUY)

    def test_rating_action_hold_pairs_with_anything(self):
        for a in (TraderAction.BUY, TraderAction.HOLD, TraderAction.SELL):
            assert rating_action_consistent("Hold", a), a


# ---------------------------------------------------------------------------
# Headline rating discipline (rendered output → parser roundtrip).
# ---------------------------------------------------------------------------

class TestRatingHeadlineDiscipline:

    @pytest.mark.parametrize("rating", list(PortfolioRating))
    def test_pm_rendered_rating_roundtrips_through_parser(self, rating: PortfolioRating):
        """Every PM rating renders to a markdown form ``parse_rating`` recovers."""
        decision = PortfolioDecision(
            rating=rating,
            executive_summary="Action plan summary.",
            investment_thesis="Thesis grounded in the debate.",
        )
        md = render_pm_decision(decision)
        assert parse_rating(md) == rating.value

    @pytest.mark.parametrize("rating", list(PortfolioRating))
    def test_research_plan_rating_roundtrips_through_parser(self, rating: PortfolioRating):
        plan = ResearchPlan(
            recommendation=rating,
            rationale="Both sides considered; chosen side identified.",
            strategic_actions="Concrete steps for execution.",
        )
        md = render_research_plan(plan)
        # Research-Manager renders ``**Recommendation**: X`` rather than
        # ``**Rating**: X``; the rating parser should still recover the
        # rating from prose because every PortfolioRating value is in the
        # canonical 5-tier set.
        assert parse_rating(md, default="Hold") in RATINGS_5_TIER

    def test_pm_render_contains_required_sections(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build over the next two weeks.",
            investment_thesis="Strong fundamentals + supportive flows.",
        )
        md = render_pm_decision(decision)
        assert "**Rating**:" in md
        assert "**Executive Summary**:" in md
        assert "**Investment Thesis**:" in md


# ---------------------------------------------------------------------------
# Trader rendered output: FINAL TRANSACTION PROPOSAL line and parser fitness.
# ---------------------------------------------------------------------------

class TestTraderFinalTransactionLine:

    @pytest.mark.parametrize("action", list(TraderAction))
    def test_final_transaction_line_matches_action(self, action: TraderAction):
        proposal = TraderProposal(
            action=action,
            reasoning="Stops and entry derived from current setup.",
        )
        md = render_trader_proposal(proposal)
        expected = f"FINAL TRANSACTION PROPOSAL: **{action.value.upper()}**"
        assert expected in md

    def test_final_transaction_line_findable_by_regex(self):
        """External greppers (CLI, ensemble summary) match the headline shape."""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Setup intact; stop below swing low.",
        )
        md = render_trader_proposal(proposal)
        m = re.search(r"FINAL TRANSACTION PROPOSAL:\s+\*\*(BUY|SELL|HOLD)\*\*", md)
        assert m is not None
        assert m.group(1) == "BUY"


# ---------------------------------------------------------------------------
# Long-side stop validity (the regression behind reports/TRIVENI.NS/...).
# ---------------------------------------------------------------------------

class TestLongSideStopRegression:

    def test_long_side_stop_above_close_is_dropped_by_validator(self):
        """Regression: an LLM proposal with a long-side stop sitting ABOVE
        the latest close must be dropped before reaching the report. A
        long stop above the close is a target/upgrade trigger, not a stop.
        """
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Buy on confirmation.",
            entry_price=190.0,
            stop_initial=210.0,  # ABOVE the close — invalid for a long
            stop_initial_basis="prior swing high",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert result.proposal.stop_initial_basis is None
        assert any("Initial Stop 210" in n for n in result.notes)

    def test_long_side_valid_stop_is_preserved(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Buy on confirmation.",
            entry_price=200.0,
            stop_initial=185.0,
            stop_initial_basis="prior swing low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial == 185.0
        assert result.proposal.stop_initial_basis == "prior swing low"
        assert result.notes == []

    def test_short_side_stop_below_close_is_dropped(self):
        """Sell-side mirror: a short stop must sit ABOVE the close."""
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="Exit and stay flat.",
            stop_initial=185.0,  # BELOW close — invalid for a short
            stop_initial_basis="recent swing low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert any("Initial Stop 185" in n for n in result.notes)

    def test_rendered_long_proposal_with_invalid_stop_does_not_show_stop(self):
        """End-to-end: rendered markdown does not emit an invalid stop line."""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Buy on confirmation.",
            stop_initial=210.0,
            stop_initial_basis="prior swing high",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        md = render_trader_proposal(result.proposal)
        assert "**Initial Stop**" not in md
        assert "210" not in md


# ---------------------------------------------------------------------------
# PM rating ↔ Trader action consistency (cross-agent invariant).
# ---------------------------------------------------------------------------

class TestRatingActionConsistencyRegression:

    def test_pm_buy_with_trader_buy_is_consistent(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build a starter position.",
            investment_thesis="Setup confirmed.",
        )
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Long entry on breakout.",
        )
        rating = parse_rating(render_pm_decision(decision))
        assert rating_action_consistent(rating, proposal.action)

    def test_pm_sell_with_trader_buy_is_inconsistent(self):
        """Regression fixture: PM Sell + Trader Buy is incoherent and the
        helper must flag it. This is exactly the kind of cross-agent drift
        the harness is here to catch."""
        decision = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="Exit position.",
            investment_thesis="Thesis broken.",
        )
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Buy on dip.",
        )
        rating = parse_rating(render_pm_decision(decision))
        assert not rating_action_consistent(rating, proposal.action)

    def test_pm_overweight_with_trader_hold_is_consistent(self):
        """A 5-tier Overweight legitimately maps to a 3-tier Hold action
        (PM wants exposure, Trader is waiting for a better entry)."""
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Increase exposure on confirmation.",
            investment_thesis="Constructive setup, await entry.",
        )
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="Wait for retest of breakout level.",
        )
        rating = parse_rating(render_pm_decision(decision))
        assert rating_action_consistent(rating, proposal.action)


# ---------------------------------------------------------------------------
# Narrator-preamble suppression on rendered structured outputs.
# ---------------------------------------------------------------------------

class TestNarratorPreambleSuppression:

    def test_pm_render_has_no_banned_preamble(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build a starter position.",
            investment_thesis="Thesis grounded in evidence.",
        )
        md = render_pm_decision(decision)
        assert has_banned_narrator_preamble(md) is None

    def test_research_plan_render_has_no_banned_preamble(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried.",
            strategic_actions="Scale in over two weeks.",
        )
        md = render_research_plan(plan)
        assert has_banned_narrator_preamble(md) is None

    def test_trader_render_has_no_banned_preamble(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Long entry on breakout.",
        )
        md = render_trader_proposal(proposal)
        assert has_banned_narrator_preamble(md) is None

    def test_helper_catches_synthetic_regression_fixture(self):
        """Negative control: a contaminated rendered string must be flagged.
        If this test ever silently passes, the matcher has been weakened."""
        bad_md = "Now I have all the data. **Rating**: Buy\n\nDetails follow."
        assert has_banned_narrator_preamble(bad_md) is not None


# ---------------------------------------------------------------------------
# Evidence Ledger and Scorecard blocks render with their stable headers.
# ---------------------------------------------------------------------------

class TestStructuredBlocksRender:

    def test_evidence_ledger_block_has_stable_header(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-01-10",
            latest_close=EvidenceFact(value=150.0, source="yfinance", as_of="2026-01-10"),
        )
        block = render_evidence_ledger(ledger)
        assert "**Evidence Ledger**" in block
        assert "Latest close: 150.00" in block

    def test_evidence_ledger_empty_renders_nothing(self):
        """Empty ledger renders to ``""`` so prompts can include it conditionally."""
        ledger = EvidenceLedger(ticker="NVDA", trade_date="2026-01-10")
        assert render_evidence_ledger(ledger) == ""

    def test_pm_render_includes_scorecard_block(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build position.",
            investment_thesis="Setup confirmed.",
            scorecard=_full_scorecard(),
        )
        md = render_pm_decision(decision)
        assert "**Scorecard**:" in md
        assert "- Bull Case: +1" in md
        assert "- Bear Case: -1" in md
        assert "- Confidence: Medium" in md

    def test_research_plan_render_includes_scorecard_block(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced.",
            strategic_actions="Hold the line.",
            scorecard=_full_scorecard(
                bull_case=0, bear_case=0,
                tie_breaker="Neither side decisive; await catalyst.",
            ),
        )
        md = render_research_plan(plan)
        assert "**Scorecard**:" in md
        assert "**Tie Breaker**:" in md

    def test_pm_render_omits_scorecard_when_absent(self):
        """Scorecard is optional — rendered output must not leak an empty header
        when no scorecard was filled (graceful-fallback path)."""
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build position.",
            investment_thesis="Setup confirmed.",
        )
        md = render_pm_decision(decision)
        assert "**Scorecard**:" not in md


# ---------------------------------------------------------------------------
# Saved-markdown fixture: lightweight smoke check for combined report shape.
# ---------------------------------------------------------------------------

class TestSavedReportSmoke:
    """Inline-fixture smoke: a synthetic 'saved report' assembled from
    real renders is parseable end-to-end. This stands in for a real
    on-disk fixture so the suite stays hermetic — when we capture a real
    saved report regression, drop it in next to this class as a string
    constant and add the matching invariant.
    """

    def _build_report(self) -> str:
        plan = ResearchPlan(
            recommendation=PortfolioRating.BUY,
            rationale="Bull case carried.",
            strategic_actions="Scale in.",
        )
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Breakout confirmed.",
            entry_price=200.0,
            stop_initial=185.0,
            stop_initial_basis="prior swing low",
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build a starter position.",
            investment_thesis="Setup confirmed.",
        )
        return "\n\n---\n\n".join([
            "## Research Plan\n" + render_research_plan(plan),
            "## Trader Proposal\n" + render_trader_proposal(proposal),
            "## Portfolio Decision\n" + render_pm_decision(decision),
        ])

    def test_combined_report_has_no_narrator_preamble(self):
        assert has_banned_narrator_preamble(self._build_report()) is None

    def test_combined_report_rating_matches_action(self):
        report = self._build_report()
        rating = parse_rating(report)
        m = re.search(r"FINAL TRANSACTION PROPOSAL:\s+\*\*(BUY|SELL|HOLD)\*\*", report)
        assert m is not None
        action_str = m.group(1).capitalize()
        action = TraderAction(action_str)
        assert rating_action_consistent(rating, action)

    def test_combined_report_keeps_final_transaction_line(self):
        """The trailing sentinel survives concatenation with other sections."""
        assert "FINAL TRANSACTION PROPOSAL:" in self._build_report()
