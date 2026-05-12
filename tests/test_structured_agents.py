"""Tests for structured-output agents (Trader and Research Manager).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader and Research Manager so all three
decision-making agents share the same shape.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
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
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.evidence_ledger import (
    EvidenceFact,
    EvidenceLedger,
)


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            entry_basis="50-DMA cluster",
            stop_initial=178.0,
            stop_initial_basis="just below 20-day low",
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5 — _50-DMA cluster_" in md
        assert "**Initial Stop**: 178.0 — _just below 20-day low_" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_dual_stop_concept_renders(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="VCP breakout in stage 2.",
            entry_price=200.0,
            entry_basis="cup-and-handle pivot",
            stop_initial=185.0,
            stop_initial_basis="just below 20-day low",
            stop_trailing=192.0,
            stop_trailing_basis="Chandelier Exit (ATR, dynamic)",
        )
        md = render_trader_proposal(p)
        assert "**Initial Stop**: 185.0 — _just below 20-day low_" in md
        assert "**Trailing Stop**: 192.0 — _Chandelier Exit (ATR, dynamic)_" in md

    def test_basis_omitted_when_absent_keeps_value_only(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=189.5,
            stop_initial=178.0,
        )
        md = render_trader_proposal(p)
        assert "**Entry Price**: 189.5\n" in md or md.endswith("**Entry Price**: 189.5")
        assert "**Initial Stop**: 178.0" in md
        assert "—" not in md.split("Entry Price")[1].split("\n")[0]

    def test_optional_fields_omitted_when_absent(self):
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        assert "Entry Price" not in md
        assert "Initial Stop" not in md
        assert "Trailing Stop" not in md
        assert "Position Sizing" not in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Evidence Scorecard (gly.4)
# ---------------------------------------------------------------------------


def _full_scorecard(**overrides) -> EvidenceScorecard:
    base = dict(
        bull_case=2,
        bear_case=-1,
        trend_technical=2,
        fundamental_quality=1,
        liquidity_risk=0,
        catalyst_clarity=1,
        macro_regime=1,
        valuation=0,
        confidence=Confidence.HIGH,
        bull_case_rationale=(
            "AI capex orders +38% QoQ per the supplier guide on 2026-04-22."
        ),
        bear_case_rationale=(
            "Channel-check shows hyperscaler order pause risk into Q3 2026."
        ),
        trend_technical_rationale=(
            "Holds 50-DMA cluster at 178 with RVOL 1.6x on the May 1 breakout."
        ),
        fundamental_quality_rationale=(
            "Gross margin 74.6% vs sector 52%; FCF conversion 92% in FY25."
        ),
        liquidity_risk_rationale=(
            "Net cash $34B vs debt $11B; cash-pile fully covers two years of capex."
        ),
        catalyst_clarity_rationale=(
            "Earnings 2026-05-28 and GTC keynote 2026-06-15 land within horizon."
        ),
        macro_regime_rationale=(
            "Semis sector RS rank 92 against SPY; FII flows positive 6 weeks running."
        ),
        valuation_rationale=(
            "FY27 PE 28x vs 5-yr median 32x; PEG 1.1 at consensus growth 26%."
        ),
        rating_rationale=(
            "Bull case strongly outweighs bear; trend and catalysts align."
        ),
        invalidating_evidence=(
            "Loss of 50-DMA on volume, or a guidance cut at next earnings."
        ),
    )
    base.update(overrides)
    return EvidenceScorecard(**base)


@pytest.mark.unit
class TestEvidenceScorecard:
    def test_full_scorecard_constructs(self):
        sc = _full_scorecard()
        assert sc.bull_case == 2
        assert sc.confidence == Confidence.HIGH
        assert sc.tie_breaker is None  # optional

    def test_score_range_minus_two_to_plus_two_is_enforced(self):
        with pytest.raises(Exception):
            _full_scorecard(bull_case=3)
        with pytest.raises(Exception):
            _full_scorecard(bear_case=-3)

    def test_tie_breaker_optional_and_renders_when_present(self):
        sc = _full_scorecard(tie_breaker="Bull and bear cases offset; awaiting catalyst.")
        assert sc.tie_breaker is not None


@pytest.mark.unit
class TestEvidenceScorecardPerCategoryRationale:
    """zlw — per-category rationale is required so divergence between runs is
    auditable. A reviewer must be able to tell whether two models disagreed on
    the *fact* driving a category score, not just on the score itself.
    """

    _CATEGORY_RATIONALE_FIELDS = (
        "bull_case_rationale",
        "bear_case_rationale",
        "trend_technical_rationale",
        "fundamental_quality_rationale",
        "liquidity_risk_rationale",
        "catalyst_clarity_rationale",
        "macro_regime_rationale",
        "valuation_rationale",
    )

    def test_all_eight_rationales_are_required_string_fields(self):
        sc = _full_scorecard()
        for field in self._CATEGORY_RATIONALE_FIELDS:
            value = getattr(sc, field)
            assert isinstance(value, str)
            assert len(value) >= 10

    def test_missing_any_rationale_fails_validation(self):
        # Drop one rationale at a time; each omission must fail Pydantic.
        for field in self._CATEGORY_RATIONALE_FIELDS:
            with pytest.raises(Exception):
                _full_scorecard(**{field: None})

    def test_rationale_below_min_length_fails_validation(self):
        with pytest.raises(Exception):
            _full_scorecard(bull_case_rationale="too short")  # < 10 chars

    def test_rationale_above_max_length_fails_validation(self):
        with pytest.raises(Exception):
            _full_scorecard(bull_case_rationale="x" * 201)  # > 200 chars

    def test_renderer_inlines_each_category_rationale(self):
        sc = _full_scorecard(
            bull_case_rationale="AI capex orders +38% QoQ per supplier guide.",
            liquidity_risk_rationale="Net cash $34B vs debt $11B (fortress).",
        )
        from tradingagents.agents.schemas import render_evidence_scorecard

        md = render_evidence_scorecard(sc)
        # Inline _rationale_ next to each category line.
        assert "Bull Case: +2" in md and "AI capex orders +38% QoQ" in md
        assert "Liquidity / Risk: 0" in md and "Net cash $34B vs debt $11B" in md
        # The signed score and the rationale appear on the same line for each
        # category, so cross-run scorecard diffs surface fact-level divergence
        # rather than just numeric divergence.
        for line in md.splitlines():
            if line.startswith("- Bull Case:"):
                assert "AI capex orders" in line
            if line.startswith("- Liquidity / Risk:"):
                assert "Net cash" in line


@pytest.mark.unit
class TestRenderResearchPlanWithScorecard:
    def test_rendered_plan_includes_scorecard_section(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="r",
            strategic_actions="s",
            scorecard=_full_scorecard(),
        )
        md = render_research_plan(plan)
        assert "**Scorecard**" in md
        assert "Bull Case: +2" in md
        assert "Bear Case: -1" in md
        assert "Confidence: High" in md
        assert "Invalidating Evidence" in md

    def test_rendered_plan_omits_scorecard_when_absent(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="r",
            strategic_actions="s",
        )
        md = render_research_plan(plan)
        assert "Scorecard" not in md

    def test_hold_with_tie_breaker_renders_tie_breaker_line(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="r",
            strategic_actions="s",
            scorecard=_full_scorecard(
                bull_case=1,
                bear_case=-1,
                tie_breaker="Bull/bear cases evenly weighted; awaiting confirmation.",
            ),
        )
        md = render_research_plan(plan)
        assert "Tie Breaker" in md
        assert "evenly weighted" in md

    def test_all_five_ratings_render_with_scorecard(self):
        for rating in PortfolioRating:
            plan = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
                scorecard=_full_scorecard(),
            )
            md = render_research_plan(plan)
            assert f"**Recommendation**: {rating.value}" in md
            assert "**Scorecard**" in md


@pytest.mark.unit
class TestRenderPMDecisionWithScorecard:
    def test_rendered_decision_includes_scorecard_section(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="e",
            investment_thesis="t",
            scorecard=_full_scorecard(),
        )
        md = render_pm_decision(decision)
        assert "**Scorecard**" in md
        assert "Confidence: High" in md

    def test_parser_compatibility_rating_line_still_present(self):
        # SignalProcessor's parser reads "**Rating**: X" — the scorecard
        # block must not displace or duplicate this line.
        decision = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="e",
            investment_thesis="t",
            scorecard=_full_scorecard(),
        )
        md = render_pm_decision(decision)
        assert md.count("**Rating**: Sell") == 1


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            entry_basis="50-DMA cluster",
            stop_initial=178.0,
            stop_initial_basis="just below 20-day low",
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5 — _50-DMA cluster_" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Research plan" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response

    def test_invalid_long_trailing_stop_is_dropped_with_validation_note(self):
        # Reproduces the TRIVENI-style failure: a Hold proposal with a
        # trailing stop above current price (a target masquerading as a
        # stop). The validator should drop the bad row and append a note
        # rather than render the misleading line verbatim.
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="Letting the breakout confirm before adding.",
            stop_trailing=435.0,
            stop_trailing_basis="20-day high — breakout confirmation level",
        )
        llm = _structured_trader_llm(captured, proposal)
        state = {
            "company_of_interest": "TRIVENI.NS",
            "investment_plan": "**Recommendation**: Hold\n**Rationale**: ...\n**Strategic Actions**: ...",
            "key_levels": "Latest close: 403.40\n50-DMA: 395.00",
        }
        trader = create_trader(llm)
        result = trader(state)
        plan = result["trader_investment_plan"]
        assert "**Trailing Stop**: 435.0" not in plan
        assert "**Validation Notes**" in plan
        assert "435" in plan and "Trailing Stop" in plan

    def test_evidence_ledger_block_appears_in_prompt_when_state_has_ledger(self):
        # When the state carries a populated EvidenceLedger, the trader's
        # user message should include the rendered ledger block instead
        # of (or alongside) the legacy key_levels string. This is the
        # consumer-side wiring of the gly.2 seam.
        captured = {}
        llm = _structured_trader_llm(captured)
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.50, source="yfinance"),
            sma_50=EvidenceFact(value=190.00, source="yfinance"),
        )
        state = {
            "company_of_interest": "NVDA",
            "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
            "evidence_ledger": ledger,
        }
        trader = create_trader(llm)
        trader(state)
        prompt = captured["prompt"]
        assert any("Evidence Ledger" in m["content"] for m in prompt)
        assert any("Latest close: 200.50" in m["content"] for m in prompt)

    def test_ledger_latest_close_drives_stop_validator(self):
        # The validator's directional check must see the ledger's
        # latest_close even when key_levels is absent — the ledger is
        # the canonical source going forward.
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="Letting the breakout confirm before adding.",
            stop_trailing=435.0,
            stop_trailing_basis="20-day high — breakout confirmation level",
        )
        llm = _structured_trader_llm(captured, proposal)
        ledger = EvidenceLedger(
            ticker="TRIVENI.NS",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=403.40, source="yfinance"),
        )
        state = {
            "company_of_interest": "TRIVENI.NS",
            "investment_plan": "**Recommendation**: Hold\n**Rationale**: ...\n**Strategic Actions**: ...",
            "evidence_ledger": ledger,
            # key_levels intentionally absent — ledger must be sufficient
        }
        trader = create_trader(llm)
        result = trader(state)
        plan = result["trader_investment_plan"]
        assert "**Trailing Stop**: 435.0" not in plan
        assert "**Validation Notes**" in plan

    def test_valid_long_proposal_renders_without_validation_notes(self):
        # When latest_close is known and stops are correctly placed the
        # validator must be a no-op — no Validation Notes footer.
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="VCP breakout off the 50-DMA.",
            entry_price=190.0,
            entry_basis="50-DMA",
            stop_initial=175.0,
            stop_initial_basis="just below 20-day low",
        )
        llm = _structured_trader_llm(captured, proposal)
        state = {
            "company_of_interest": "NVDA",
            "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
            "key_levels": "Latest close: 200.00\n50-DMA: 190.00",
        }
        trader = create_trader(llm)
        result = trader(state)
        plan = result["trader_investment_plan"]
        assert "**Initial Stop**: 175.0" in plan
        assert "**Validation Notes**" not in plan


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response

    def test_evidence_ledger_block_appears_in_prompt(self):
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        state = _make_rm_state()
        state["evidence_ledger"] = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.50, source="yfinance"),
        )
        rm(state)
        prompt = captured["prompt"]
        assert "Evidence Ledger" in prompt
        assert "Latest close: 200.50" in prompt

    def test_prompt_instructs_scorecard_fill(self):
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        # Each named evidence category must appear in the instructions
        # so the LLM knows what to score.
        for category in (
            "Bull Case",
            "Bear Case",
            "Trend",
            "Fundamental Quality",
            "Catalyst",
            "Macro",
            "Valuation",
        ):
            assert category in prompt, f"missing category {category!r} in RM prompt"
        # Hold requires explicit tie-breaker mention
        assert "tie" in prompt.lower() and "breaker" in prompt.lower()
        # Buy/Sell require invalidation
        assert "invalidat" in prompt.lower()


# ---------------------------------------------------------------------------
# Portfolio Manager: ledger inclusion in prompt
# ---------------------------------------------------------------------------


def _make_pm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...",
        "trader_investment_plan": "**Action**: Buy\n**Reasoning**: ...",
        "risk_debate_state": {
            "history": "Risk analyst arguments here.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "latest_speaker": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_pm_llm(captured: dict, decision: PortfolioDecision | None = None):
    if decision is None:
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold and reassess.",
            investment_thesis="Balanced view across analysts.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestPortfolioManagerAgent:
    def test_evidence_ledger_block_appears_in_prompt(self):
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm)
        state = _make_pm_state()
        state["evidence_ledger"] = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.50, source="yfinance"),
        )
        pm(state)
        prompt = captured["prompt"]
        assert "Evidence Ledger" in prompt
        assert "Latest close: 200.50" in prompt

    def test_prompt_instructs_scorecard_fill(self):
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm)
        pm(_make_pm_state())
        prompt = captured["prompt"]
        for category in (
            "Bull Case",
            "Bear Case",
            "Trend",
            "Fundamental Quality",
            "Catalyst",
            "Macro",
            "Valuation",
        ):
            assert category in prompt, f"missing category {category!r} in PM prompt"
        assert "tie" in prompt.lower() and "breaker" in prompt.lower()
        assert "invalidat" in prompt.lower()
