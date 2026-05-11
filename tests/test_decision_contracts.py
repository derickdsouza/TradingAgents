"""Tests for the decision-contract validator (TraderProposal slice).

These tests pin the semantic invariants the validator must enforce on
typed Trader output before it is rendered. The schema alone keeps the
output parseable; the validator keeps it tradeable.
"""

from datetime import datetime, timedelta

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.utils.decision_contracts import (
    PortfolioValidationContext,
    TraderValidationContext,
    ValidatedPortfolioDecision,
    render_pm_validation_notes,
    render_validation_notes,
    validate_portfolio_decision,
    validate_trader_proposal,
)


@pytest.mark.unit
class TestValidateTraderProposal:
    def test_valid_long_proposal_passes_through_unchanged(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=190.0,
            entry_basis="50-DMA cluster",
            stop_initial=175.0,
            stop_initial_basis="just below 20-day low",
            stop_trailing=180.0,
            stop_trailing_basis="Chandelier Exit (ATR, dynamic)",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.notes == []
        assert result.proposal.stop_initial == 175.0
        assert result.proposal.stop_trailing == 180.0
        assert result.proposal.entry_price == 190.0

    def test_long_initial_stop_above_close_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=210.0,
            stop_initial_basis="20-day high — breakout confirmation",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert result.proposal.stop_initial_basis is None
        assert any("Initial Stop" in n and "210" in n for n in result.notes)

    def test_long_trailing_stop_above_close_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="r",
            stop_trailing=435.0,
            stop_trailing_basis="20-day high — breakout confirmation level",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=403.4)
        )
        assert result.proposal.stop_trailing is None
        assert result.proposal.stop_trailing_basis is None
        assert any("Trailing Stop" in n and "435" in n for n in result.notes)

    def test_short_initial_stop_below_close_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            stop_initial=180.0,
            stop_initial_basis="below 20-day low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert any("Initial Stop" in n for n in result.notes)

    def test_short_trailing_stop_below_close_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            stop_trailing=180.0,
            stop_trailing_basis="rising 50-DMA",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_trailing is None
        assert any("Trailing Stop" in n for n in result.notes)

    def test_long_stop_above_entry_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=190.0,
            stop_initial=195.0,  # above entry: invalid for long
            stop_initial_basis="just below 20-day low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert any("entry" in n.lower() for n in result.notes)

    def test_short_stop_below_entry_is_dropped(self):
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            entry_price=190.0,
            stop_initial=185.0,  # below entry: invalid for short
            stop_initial_basis="below 20-day low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=180.0)
        )
        assert result.proposal.stop_initial is None
        assert any("entry" in n.lower() for n in result.notes)

    def test_no_close_means_no_directional_check(self):
        # Without a known latest close we can't disprove the value;
        # pass-through preserves whatever the model emitted.
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=210.0,
            stop_initial_basis="x",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=None)
        )
        assert result.proposal.stop_initial == 210.0
        assert result.notes == []

    def test_basis_without_value_is_dropped(self):
        # An orphan basis (basis populated, value missing) is a render-time
        # bug because the renderer skips the field entirely; defensively
        # strip the orphan basis so it never leaks into a future renderer.
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=None,
            stop_initial_basis="just below 20-day low",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial_basis is None
        assert any("orphan" in n.lower() or "basis" in n.lower() for n in result.notes)

    def test_dropping_value_also_drops_basis(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=210.0,
            stop_initial_basis="20-day high — breakout confirmation",
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.proposal.stop_initial is None
        assert result.proposal.stop_initial_basis is None

    def test_changed_property_reflects_notes(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=210.0,
        )
        result = validate_trader_proposal(
            proposal, TraderValidationContext(latest_close=200.0)
        )
        assert result.changed is True

        proposal_ok = TraderProposal(action=TraderAction.HOLD, reasoning="r")
        result_ok = validate_trader_proposal(
            proposal_ok, TraderValidationContext(latest_close=200.0)
        )
        assert result_ok.changed is False


@pytest.mark.unit
class TestRenderValidationNotes:
    def test_empty_notes_render_empty_string(self):
        assert render_validation_notes([]) == ""

    def test_notes_render_as_markdown_list(self):
        md = render_validation_notes(["First note", "Second note"])
        assert "**Validation Notes**" in md
        assert "- First note" in md
        assert "- Second note" in md


# ---------------------------------------------------------------------------
# Portfolio Manager decision-contract validator.
#
# The Trader validator above enforces structural invariants on stops; the
# Portfolio Manager validator enforces directional coherence on the
# ``price_target_horizon`` field. A Hold call with a -13% implied return
# (the PARACABLES regression) is structurally a Sell signal — the schema
# permits the number but the validator drops it loudly.
# ---------------------------------------------------------------------------


_TRADE_DATE = datetime(2026, 5, 11)


def _pm_decision(**overrides) -> PortfolioDecision:
    """Build a minimal PortfolioDecision; callers override only what they test."""
    base = dict(
        rating=PortfolioRating.HOLD,
        executive_summary="Hold position, await catalyst.",
        investment_thesis="Evidence balanced; no decisive driver.",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


@pytest.mark.unit
class TestPortfolioValidationContext:
    def test_context_has_required_fields(self):
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
        )
        assert ctx.trade_date == _TRADE_DATE
        assert ctx.latest_close == 100.0
        assert ctx.close_as_of == _TRADE_DATE

    def test_context_optional_close_fields_default_none(self):
        ctx = PortfolioValidationContext(trade_date=_TRADE_DATE)
        assert ctx.latest_close is None
        assert ctx.close_as_of is None


@pytest.mark.unit
class TestValidatedPortfolioDecisionDataclass:
    def test_changed_property_reflects_notes(self):
        decision = _pm_decision()
        v = ValidatedPortfolioDecision(decision=decision, notes=["x"])
        assert v.changed is True
        v_clean = ValidatedPortfolioDecision(decision=decision, notes=[])
        assert v_clean.changed is False


@pytest.mark.unit
class TestValidatePortfolioDecisionPassthrough:
    def test_no_target_passes_unchanged(self):
        decision = _pm_decision(rating=PortfolioRating.BUY)
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.notes == []
        assert result.decision.price_target_horizon is None


@pytest.mark.unit
class TestPortfolioContractRule1DirectionalContract:
    """Rule: for Buy/Overweight target must be meaningfully above close;
    for Sell/Underweight meaningfully below; for Hold within ±10% band
    (fixed placeholder for Slice 1; Slice 2 replaces with vol-scaled)."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
        )

    def test_buy_target_below_close_is_dropped(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY, price_target_horizon=95.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any(
            "TARGET_DIRECTION_VIOLATION" in n or "below close" in n.lower()
            for n in result.notes
        )

    def test_sell_target_above_close_is_dropped(self):
        decision = _pm_decision(
            rating=PortfolioRating.SELL, price_target_horizon=110.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_overweight_target_5pct_above_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT, price_target_horizon=105.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 105.0
        assert result.decision.target_basis == "DCF"
        assert result.notes == []

    def test_underweight_target_5pct_below_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.UNDERWEIGHT, price_target_horizon=95.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 95.0
        assert result.notes == []

    def test_hold_target_3pct_above_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.HOLD, price_target_horizon=103.0,
            target_basis="range midpoint",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 103.0
        assert result.notes == []

    def test_hold_target_below_close_within_band_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.HOLD, price_target_horizon=97.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 97.0
        assert result.notes == []

    def test_hold_paracables_regression_target_13pct_below_is_dropped(self):
        """The PARACABLES bug: Hold + target = -13% vs close → dropped."""
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=59.77,
            close_as_of=_TRADE_DATE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD, price_target_horizon=52.0,
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.decision.price_target_horizon is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_hold_target_far_above_band_is_dropped(self):
        """Hold + target +20% (outside ±10% band) → dropped."""
        decision = _pm_decision(
            rating=PortfolioRating.HOLD, price_target_horizon=120.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_buy_target_within_epsilon_is_dropped(self):
        """Buy + target only +1% above close (within epsilon=0.03) → dropped.
        The target is not meaningfully above close — it is statistical noise."""
        decision = _pm_decision(
            rating=PortfolioRating.BUY, price_target_horizon=101.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)


@pytest.mark.unit
class TestPortfolioContractRule2LoudFailMissingClose:
    """Rule: opposite of Trader behaviour — missing/stale close MUST drop
    the target rather than silently skip. Without a reference price the
    target is unverifiable and we refuse to publish unverifiable numbers."""

    def test_missing_close_drops_target_with_loud_note(self):
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=None, close_as_of=None,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY, price_target_horizon=120.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any("LATEST_CLOSE_STALE" in n for n in result.notes)

    def test_stale_close_drops_target_with_loud_note(self):
        """close_as_of more than 1 calendar day before trade_date is stale."""
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE - timedelta(days=3),
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY, price_target_horizon=120.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.decision.price_target_horizon is None
        assert any("LATEST_CLOSE_STALE" in n for n in result.notes)

    def test_close_within_one_session_is_not_stale(self):
        """close_as_of 1 day before trade_date is fresh enough."""
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE - timedelta(days=1),
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY, price_target_horizon=120.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.decision.price_target_horizon == 120.0
        assert result.notes == []


@pytest.mark.unit
class TestPortfolioContractRule3OrphanBasis:
    def test_orphan_basis_without_target_is_cleared(self):
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=None,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, ctx)
        assert result.decision.target_basis is None
        assert any("TARGET_ORPHAN_BASIS" in n for n in result.notes)


@pytest.mark.unit
class TestPortfolioSchemaMigration:
    """Legacy memory-log entries (and existing tests) used ``price_target``.
    The new schema must accept the old key and coerce it to
    ``price_target_horizon`` so old logs still load."""

    def test_legacy_price_target_key_is_coerced(self):
        decision = PortfolioDecision.model_validate({
            "rating": "Hold",
            "executive_summary": "summary",
            "investment_thesis": "thesis",
            "price_target": 100.0,
        })
        assert decision.price_target_horizon == 100.0

    def test_legacy_kwarg_constructor_still_works(self):
        """Existing tests instantiate via ``price_target=215.0`` kwarg."""
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target=215.0,
        )
        assert decision.price_target_horizon == 215.0

    def test_new_name_takes_precedence_when_both_supplied(self):
        decision = PortfolioDecision.model_validate({
            "rating": "Hold",
            "executive_summary": "s",
            "investment_thesis": "t",
            "price_target": 100.0,
            "price_target_horizon": 120.0,
        })
        assert decision.price_target_horizon == 120.0


@pytest.mark.unit
class TestRenderPmDecisionDualLabel:
    """Until Slice 2, render BOTH ``Price Target`` and ``Horizon Target``
    labels for the same value so downstream consumers can migrate."""

    def test_both_labels_emitted_when_target_set(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
        )
        md = render_pm_decision(decision)
        assert "**Price Target**: 120.0" in md
        assert "**Horizon Target**: 120.0" in md

    def test_neither_label_emitted_when_target_absent(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
        )
        md = render_pm_decision(decision)
        assert "**Price Target**" not in md
        assert "**Horizon Target**" not in md


@pytest.mark.unit
class TestRenderPmValidationNotes:
    def test_empty_notes_render_empty_string(self):
        assert render_pm_validation_notes([]) == ""

    def test_notes_render_as_markdown_list(self):
        md = render_pm_validation_notes(["First note", "Second note"])
        assert "**Validation Notes**" in md
        assert "- First note" in md
        assert "- Second note" in md
