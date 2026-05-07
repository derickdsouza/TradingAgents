"""Tests for the decision-contract validator (TraderProposal slice).

These tests pin the semantic invariants the validator must enforce on
typed Trader output before it is rendered. The schema alone keeps the
output parseable; the validator keeps it tradeable.
"""

import pytest

from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.agents.utils.decision_contracts import (
    TraderValidationContext,
    render_validation_notes,
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
