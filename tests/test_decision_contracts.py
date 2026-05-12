"""Tests for the decision-contract validator (TraderProposal slice).

These tests pin the semantic invariants the validator must enforce on
typed Trader output before it is rendered. The schema alone keeps the
output parseable; the validator keeps it tradeable.
"""

from datetime import datetime, timedelta
from typing import Optional

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
class TestStopBufferRule:
    """74l — a stop placed AT a cited structural support (200-DMA, 50-DMA,
    52w-low, 20d-low) is triggered by routine retests of that level. The
    validator flags any stop sitting within ``buffer_pct`` of a cited
    support so the reader sees the wedge-stop risk explicitly. The flag is
    advisory (no drop) — the trader's basis label may still be defensible.
    Buffer thresholds scale with horizon: swing 3%, position 4%, long-term
    5% (wider holds tolerate wider whipsaws).
    """

    def _ctx(self, **kw):
        kw.setdefault("latest_close", 40.0)
        return TraderValidationContext(**kw)

    def test_stop_at_cited_support_emits_no_buffer_note(self):
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=38.0,
            stop_initial=36.47,  # exactly the 200-DMA
            stop_initial_basis="200-DMA",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47),),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n and "200-DMA" in n for n in result.notes)
        # Advisory only — the stop itself is preserved.
        assert result.proposal.stop_initial == 36.47

    def test_stop_with_5pct_buffer_below_support_is_clean(self):
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=38.0,
            stop_initial=34.50,  # 200-DMA (36.47) minus ~5.4%
            stop_initial_basis="200-DMA −5%",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47),),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert not any(STOP_NO_BUFFER in n for n in result.notes)

    def test_stop_just_above_support_also_fires(self):
        """Stop above cited support means the level hasn't been tested yet
        on the way down — a single retest takes the trade out before the
        thesis is invalidated. Worse than a stop AT the level."""
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=38.0,
            stop_initial=37.10,  # just above the 200-DMA at 36.47 (≈1.7% above)
            stop_initial_basis="thesis-break",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47),),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n for n in result.notes)

    def test_trailing_stop_at_support_also_flags(self):
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_trailing=50.05,  # right at the 50-DMA
            stop_trailing_basis="rising 50-DMA",
        )
        ctx = self._ctx(
            latest_close=55.0,
            support_levels=(("50-DMA", 50.0),),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n and "50-DMA" in n for n in result.notes)
        # Trailing stop preserved (advisory only).
        assert result.proposal.stop_trailing == 50.05

    def test_position_horizon_uses_4pct_buffer(self):
        """At position horizon, 3.5% buffer is still too tight (4% needed)."""
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=35.20,  # ~3.5% below 36.47
            stop_initial_basis="thesis-break",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47),),
            horizon="position",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n for n in result.notes)

    def test_long_term_horizon_uses_5pct_buffer(self):
        """At long-term horizon, even 4.5% below is still too tight."""
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=34.83,  # ~4.5% below 36.47
            stop_initial_basis="thesis-break",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47),),
            horizon="long-term",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n for n in result.notes)

    def test_no_support_levels_no_check(self):
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=36.47,
            stop_initial_basis="200-DMA",
        )
        # Backward compat: callers that don't supply support_levels skip
        # the rule entirely.
        ctx = TraderValidationContext(latest_close=40.0)
        result = validate_trader_proposal(proposal, ctx)
        assert not any(STOP_NO_BUFFER in n for n in result.notes)

    def test_short_side_stop_at_resistance_also_flags(self):
        """For Sell-side trades, the stop sits ABOVE close at a cited
        *resistance*. Same buffer logic applies symmetrically."""
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            stop_initial=200.10,  # at 200.0 resistance
            stop_initial_basis="prior swing high",
        )
        ctx = TraderValidationContext(
            latest_close=180.0,
            support_levels=(("prior swing high", 200.0),),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        assert any(STOP_NO_BUFFER in n for n in result.notes)

    def test_multiple_supports_each_emits_distinct_note(self):
        """If a stop is wedged near *two* cited supports the validator
        flags both so the trader sees the full picture."""
        from tradingagents.agents.utils.decision_contracts import STOP_NO_BUFFER

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            stop_initial=36.45,
            stop_initial_basis="200-DMA + 52w-low cluster",
        )
        ctx = self._ctx(
            support_levels=(("200-DMA", 36.47), ("52w-low", 36.30)),
            horizon="swing",
        )
        result = validate_trader_proposal(proposal, ctx)
        no_buffer_notes = [n for n in result.notes if STOP_NO_BUFFER in n]
        assert len(no_buffer_notes) == 2


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
        # Slice 2: provide annualised_volatility so the Hold band is
        # calibrated and the HOLD_BAND_UNCALIBRATED informational note
        # doesn't fire. Volatility of 0.20 → 0.5*0.20 = 0.10 band → same
        # bounds as Slice 1's fixed ±10% band, preserving these test
        # assertions verbatim.
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
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
        # Slice 3: target_basis is normalised to canonical lowercase.
        assert result.decision.target_basis == "dcf"
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
        # Slice 3: ``range midpoint`` is not in the controlled vocabulary;
        # use the canonical ``range_bound`` token.
        decision = _pm_decision(
            rating=PortfolioRating.HOLD, price_target_horizon=103.0,
            target_basis="range_bound",
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


# ---------------------------------------------------------------------------
# Slice 2: schema additions, currency derivation, volatility-scaled Hold band,
# banned-placeholder rule, currency-mismatch rule, expected-return render.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPortfolioDecisionTargetCurrencyField:
    """Slice 2: PortfolioDecision must accept an optional target_currency."""

    def test_target_currency_field_defaults_to_none(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
        )
        assert decision.target_currency is None

    def test_target_currency_field_accepts_iso_4217(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
            target_currency="INR",
        )
        assert decision.target_currency == "INR"


@pytest.mark.unit
class TestPortfolioValidationContextSlice2Fields:
    """Slice 2 extends the context with vol, close_currency, hold-band default."""

    def test_new_fields_default_to_sensible_values(self):
        ctx = PortfolioValidationContext(trade_date=_TRADE_DATE)
        assert ctx.annualised_volatility is None
        assert ctx.close_currency is None
        assert ctx.hold_band_pct_default == 0.10

    def test_new_fields_accept_explicit_values(self):
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.40,
            close_currency="INR",
            hold_band_pct_default=0.08,
        )
        assert ctx.annualised_volatility == 0.40
        assert ctx.close_currency == "INR"
        assert ctx.hold_band_pct_default == 0.08


@pytest.mark.unit
class TestCurrencyFromTicker:
    """Ticker-suffix → ISO-4217 currency derivation."""

    def test_nse_suffix_is_inr(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("PARACABLES.NS") == "INR"

    def test_bse_suffix_is_inr(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("RELIANCE.BO") == "INR"

    def test_london_suffix_is_gbp(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("BARC.L") == "GBP"

    def test_hong_kong_suffix_is_hkd(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("0700.HK") == "HKD"

    def test_toronto_suffix_is_cad(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("SHOP.TO") == "CAD"

    def test_sydney_suffix_is_aud(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("BHP.AX") == "AUD"

    def test_tokyo_suffix_is_jpy(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("7203.T") == "JPY"

    def test_paris_suffix_is_eur(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("MC.PA") == "EUR"

    def test_amsterdam_suffix_is_eur(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("ASML.AS") == "EUR"

    def test_xetra_suffix_is_eur(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("SAP.DE") == "EUR"

    def test_no_suffix_defaults_to_usd(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("AAPL") == "USD"

    def test_unrecognised_suffix_defaults_to_usd(self):
        from tradingagents.agents.utils.decision_contracts import _currency_from_ticker
        assert _currency_from_ticker("FOO.ZZZ") == "USD"


@pytest.mark.unit
class TestPortfolioContractVolatilityScaledHoldBand:
    """Slice 2: Hold band scales with annualised volatility.

    Formula: ``hold_band_pct = max(0.05, min(0.20, 0.5 * vol * sqrt(1.0)))``.
    When vol is unavailable the band falls back to
    ``hold_band_pct_default`` and emits ``HOLD_BAND_UNCALIBRATED``.
    """

    def _ctx(self, vol=None) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=vol,
        )

    def test_high_vol_hold_target_within_scaled_band_is_preserved(self):
        # vol=0.50 → band = clamp(0.5*0.50*1.0)=0.25 → clamped to 0.20.
        # Target +18% is inside ±20% → preserved.
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=118.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx(vol=0.50))
        assert result.decision.price_target_horizon == 118.0

    def test_low_vol_hold_target_outside_scaled_band_is_dropped(self):
        # vol=0.08 → band = clamp(0.5*0.08*1.0)=0.04 → clamped up to 0.05.
        # Target +18% is far outside ±5% → dropped.
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_DIRECTION_VIOLATION,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=118.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx(vol=0.08))
        assert result.decision.price_target_horizon is None
        assert any(TARGET_DIRECTION_VIOLATION in n for n in result.notes)

    def test_no_vol_hold_target_outside_fallback_band_is_dropped_with_uncalibrated_note(self):
        # No vol → fallback ±10%. Target +12% → dropped (DIRECTION) and
        # HOLD_BAND_UNCALIBRATED note also emitted.
        from tradingagents.agents.utils.decision_contracts import (
            HOLD_BAND_UNCALIBRATED,
            TARGET_DIRECTION_VIOLATION,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=112.0,
        )
        result = validate_portfolio_decision(decision, self._ctx(vol=None))
        assert result.decision.price_target_horizon is None
        assert any(TARGET_DIRECTION_VIOLATION in n for n in result.notes)
        assert any(HOLD_BAND_UNCALIBRATED in n for n in result.notes)

    def test_no_vol_hold_target_inside_fallback_band_is_preserved_with_uncalibrated_note(self):
        # No vol + Hold + target +3% → preserved (inside ±10%), but
        # HOLD_BAND_UNCALIBRATED is still emitted to signal the band came
        # from the fallback, not from real volatility.
        from tradingagents.agents.utils.decision_contracts import (
            HOLD_BAND_UNCALIBRATED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=103.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx(vol=None))
        assert result.decision.price_target_horizon == 103.0
        assert any(HOLD_BAND_UNCALIBRATED in n for n in result.notes)


@pytest.mark.unit
class TestPortfolioContractBannedPlaceholder:
    """Slice 2 rule: target within 0.5% of close is a banned placeholder.

    Slice 3 will add an opt-out via ``target_basis == "catalyst_neutral"``
    in the controlled-vocabulary patch; Slice 2 has no opt-out (the field
    is still free-text).
    """

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_target_exactly_equal_to_close_is_dropped(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_EQUALS_CLOSE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=100.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any(TARGET_EQUALS_CLOSE in n for n in result.notes)

    def test_target_within_half_percent_of_close_is_dropped(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_EQUALS_CLOSE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=100.3,  # 0.3% above close
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert any(TARGET_EQUALS_CLOSE in n for n in result.notes)

    def test_target_2pct_above_close_passes_placeholder_rule(self):
        # 2% above close is far enough to be a real (small) target. Hold +
        # 2% sits inside the calibrated band (±10% at vol=0.20) so this is
        # also preserved by the directional rule.
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=102.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 102.0
        assert result.notes == []


@pytest.mark.unit
class TestPortfolioContractTargetCurrency:
    """Slice 2 rule: target_currency must match close_currency.

    When the LLM names a currency that does not match the instrument's
    quote currency, the target is meaningless — a $120 target on an INR-
    denominated stock is structurally a translation bug, not a horizon
    level.
    """

    def _ctx(self, close_ccy: Optional[str] = "INR") -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
            close_currency=close_ccy,
        )

    def test_mismatched_currency_drops_target(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_CURRENCY_MISMATCH,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency="USD",
        )
        result = validate_portfolio_decision(decision, self._ctx(close_ccy="INR"))
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any(TARGET_CURRENCY_MISMATCH in n for n in result.notes)

    def test_matched_currency_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency="INR",
        )
        result = validate_portfolio_decision(decision, self._ctx(close_ccy="INR"))
        assert result.decision.price_target_horizon == 120.0
        assert result.decision.target_currency == "INR"
        assert result.notes == []

    def test_none_target_currency_is_not_a_mismatch(self):
        # When the LLM doesn't fill target_currency, the validator must
        # not treat None as a mismatch — Slice 2 leaves currency-defaulting
        # to the caller (portfolio_manager), not the validator.
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency=None,
        )
        result = validate_portfolio_decision(decision, self._ctx(close_ccy="INR"))
        assert result.decision.price_target_horizon == 120.0
        assert result.notes == []

    def test_none_close_currency_is_not_a_mismatch(self):
        # When close_currency isn't known we can't disprove the target's
        # currency. Pass-through with no note (mirrors the
        # latest_close=None gating elsewhere).
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency="USD",
        )
        result = validate_portfolio_decision(decision, self._ctx(close_ccy=None))
        assert result.decision.price_target_horizon == 120.0
        assert result.notes == []


@pytest.mark.unit
class TestRenderExpectedReturn:
    """Slice 2: render an ``**Expected Return**`` line under the target
    pair when both ``price_target_horizon`` and ``latest_close`` are
    known. Signed percentage rounded to one decimal."""

    def test_positive_expected_return_is_rendered(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=110.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Expected Return**: +10.0%" in md

    def test_negative_expected_return_is_rendered(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=90.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Expected Return**: -10.0%" in md

    def test_no_close_means_no_expected_return_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=110.0,
        )
        md = render_pm_decision(decision)
        assert "**Expected Return**" not in md

    def test_no_target_means_no_expected_return_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Expected Return**" not in md

    def test_expected_return_appears_below_horizon_target(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=110.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        # Horizon Target must appear before Expected Return in the output.
        assert md.index("**Horizon Target**") < md.index("**Expected Return**")


@pytest.mark.unit
class TestPortfolioManagerSlice2Wiring:
    """The portfolio_manager node must wire the Slice 2 inputs:

    - close_currency defaulted from the ticker suffix
    - target_currency defaulted to close_currency when None
    - annualised_volatility sourced from the evidence ledger's `extras`
    - latest_close threaded into render_pm_decision so Expected Return
      appears in the rendered markdown
    """

    def _state(self, *, ticker: str, latest_close: float, vol: Optional[float] = None,
               trade_date: str = "2026-05-11"):
        from unittest.mock import MagicMock
        from tradingagents.agents.utils.evidence_ledger import (
            EvidenceFact,
            EvidenceLedger,
        )
        extras = {}
        if vol is not None:
            extras["annualised_volatility"] = EvidenceFact(
                value=vol, source="market_analyst", as_of=trade_date,
            )
        ledger = EvidenceLedger(
            ticker=ticker,
            trade_date=trade_date,
            latest_close=EvidenceFact(
                value=latest_close, source="yfinance", as_of=trade_date,
            ),
            extras=extras,
        )
        return {
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "past_context": "",
            "evidence_ledger": ledger,
            "risk_debate_state": {
                "history": "h",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "judge_decision": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "count": 1,
            },
            "investment_plan": "rp",
            "trader_investment_plan": "tp",
        }

    def _llm(self, decision: PortfolioDecision):
        from unittest.mock import MagicMock
        structured = MagicMock()
        structured.invoke.return_value = decision
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        return llm

    def test_render_includes_expected_return_for_indian_ticker(self):
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Accumulate gradually on confirmation.",
            investment_thesis="Setup intact; institutional flows constructive.",
            price_target_horizon=120.0,
            target_basis="DCF",
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="PARACABLES.NS", latest_close=100.0, vol=0.30))
        md = result["final_trade_decision"]
        assert "**Expected Return**: +20.0%" in md

    def test_indian_ticker_default_target_currency_matches_close_currency(self):
        # Decision has no explicit target_currency; the wiring must default
        # it to the close currency (INR for .NS) so the validator does NOT
        # drop the target with a currency mismatch.
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency=None,
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="PARACABLES.NS", latest_close=100.0, vol=0.30))
        md = result["final_trade_decision"]
        assert "**Horizon Target**: 120.0" in md
        assert "TARGET_CURRENCY_MISMATCH" not in md

    def test_us_ticker_with_inr_target_currency_is_rejected(self):
        # AAPL → close_currency=USD. target_currency=INR → mismatch → drop.
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
            target_basis="DCF",
            target_currency="INR",
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="AAPL", latest_close=100.0, vol=0.30))
        md = result["final_trade_decision"]
        assert "**Horizon Target**: 120.0" not in md
        assert "TARGET_CURRENCY_MISMATCH" in md

    def test_ledger_annualised_volatility_drives_hold_band(self):
        # Hold + target +18% + vol=0.50 → band clamps to 20% → preserved.
        # Without the wiring this would fall back to ±10% and drop.
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=118.0,
            target_basis="range_bound",
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="AAPL", latest_close=100.0, vol=0.50))
        md = result["final_trade_decision"]
        assert "**Horizon Target**: 118.0" in md
        assert "HOLD_BAND_UNCALIBRATED" not in md

    def test_missing_volatility_emits_uncalibrated_note_on_hold(self):
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=103.0,
            target_basis="range_bound",
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="AAPL", latest_close=100.0, vol=None))
        md = result["final_trade_decision"]
        # Target inside ±10% so preserved, but informational note fires.
        assert "**Horizon Target**: 103.0" in md
        assert "HOLD_BAND_UNCALIBRATED" in md


# ---------------------------------------------------------------------------
# Slice 3: rating_target_disagreement enum, controlled-vocab bases,
# pullback fields. Preserves contrarian PMs and pullback-aware Holds
# without weakening the directional contract.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRatingTargetDisagreementSchema:
    """Slice 3: PortfolioDecision must accept the disagreement enum field."""

    def test_disagreement_field_defaults_to_none(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
        )
        assert decision.rating_target_disagreement is None

    def test_disagreement_field_accepts_enum_value(self):
        from tradingagents.agents.schemas import RatingTargetDisagreement
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            rating_target_disagreement=RatingTargetDisagreement.DIVIDEND_FLOOR,
        )
        assert decision.rating_target_disagreement == RatingTargetDisagreement.DIVIDEND_FLOOR

    def test_disagreement_field_accepts_string_value(self):
        from tradingagents.agents.schemas import RatingTargetDisagreement
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            rating_target_disagreement="quality_premium",
        )
        assert decision.rating_target_disagreement == RatingTargetDisagreement.QUALITY_PREMIUM


@pytest.mark.unit
class TestPullbackZoneSchema:
    """Slice 3: PortfolioDecision must accept pullback_zone + pullback_basis."""

    def test_pullback_fields_default_to_none(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
        )
        assert decision.pullback_zone is None
        assert decision.pullback_basis is None

    def test_pullback_fields_accept_values(self):
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            pullback_zone=92.0,
            pullback_basis="moving_average",
        )
        assert decision.pullback_zone == 92.0
        assert decision.pullback_basis == "moving_average"


@pytest.mark.unit
class TestRatingTargetDisagreementValidator:
    """Slice 3: setting rating_target_disagreement (non-none) suspends the
    directional contract for the decision and emits the informational note."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_hold_with_negative_target_and_dividend_floor_is_preserved(self):
        """Hold + target -13% would normally drop; with dividend_floor it survives."""
        from tradingagents.agents.schemas import RatingTargetDisagreement
        from tradingagents.agents.utils.decision_contracts import (
            RATING_TARGET_DISAGREEMENT_SET,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=87.0,  # -13% from close 100
            target_basis="dcf",
            rating_target_disagreement=RatingTargetDisagreement.DIVIDEND_FLOOR,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 87.0
        assert result.decision.target_basis == "dcf"
        assert any(RATING_TARGET_DISAGREEMENT_SET in n for n in result.notes)
        # Critical: NOT a TARGET_DIRECTION_VIOLATION drop.
        assert not any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_hold_with_negative_target_and_disagreement_none_is_still_dropped(self):
        """Setting disagreement=none does NOT suspend the contract."""
        from tradingagents.agents.schemas import RatingTargetDisagreement
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=87.0,
            target_basis="dcf",
            rating_target_disagreement=RatingTargetDisagreement.NONE,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_overweight_with_negative_target_and_quality_premium_is_preserved(self):
        from tradingagents.agents.schemas import RatingTargetDisagreement
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=97.0,  # -3% from close 100
            target_basis="dcf",
            rating_target_disagreement=RatingTargetDisagreement.QUALITY_PREMIUM,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 97.0

    def test_buy_with_negative_target_and_no_disagreement_is_dropped(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=90.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)


@pytest.mark.unit
class TestTargetBasisVocabulary:
    """Slice 3: target_basis must come from the controlled vocabulary;
    case-insensitive match coerces to canonical lowercase."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_uppercase_dcf_is_coerced_to_lowercase_and_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="DCF",  # uppercase
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 120.0
        assert result.decision.target_basis == "dcf"  # canonical

    def test_unrecognised_basis_is_dropped_but_target_preserved(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_BASIS_UNRECOGNISED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="thesis_break",  # not in vocabulary
        )
        result = validate_portfolio_decision(decision, self._ctx())
        # Target itself survives.
        assert result.decision.price_target_horizon == 120.0
        # But the bad basis is dropped.
        assert result.decision.target_basis is None
        assert any(TARGET_BASIS_UNRECOGNISED in n for n in result.notes)

    def test_none_basis_emits_no_vocabulary_note(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_BASIS_UNRECOGNISED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis=None,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert not any(TARGET_BASIS_UNRECOGNISED in n for n in result.notes)

    def test_catalyst_neutral_basis_allows_target_equal_to_close(self):
        """Slice 3 opt-out: target_basis=catalyst_neutral skips banned-placeholder."""
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=100.0,  # exactly equal to close
            target_basis="catalyst_neutral",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.price_target_horizon == 100.0
        assert result.decision.target_basis == "catalyst_neutral"
        assert not any("TARGET_EQUALS_CLOSE" in n for n in result.notes)


@pytest.mark.unit
class TestPullbackZoneDirection:
    """Slice 3: pullback_zone has its own directional rule.

    Long ratings (Buy/Overweight): pullback sits below close.
    Short ratings (Sell/Underweight): pullback sits above close.
    Hold: either side allowed (Hold-with-bullish-lean is legitimate).
    """

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_buy_with_pullback_below_close_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="dcf",
            pullback_zone=92.0,
            pullback_basis="moving_average",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_zone == 92.0
        assert result.decision.pullback_basis == "moving_average"

    def test_buy_with_pullback_above_close_is_dropped(self):
        from tradingagents.agents.utils.decision_contracts import (
            PULLBACK_DIRECTION_VIOLATION,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="dcf",
            pullback_zone=105.0,  # ABOVE close → not a pullback for a long
            pullback_basis="moving_average",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_zone is None
        assert result.decision.pullback_basis is None
        assert any(PULLBACK_DIRECTION_VIOLATION in n for n in result.notes)

    def test_sell_with_pullback_below_close_is_dropped(self):
        from tradingagents.agents.utils.decision_contracts import (
            PULLBACK_DIRECTION_VIOLATION,
        )
        decision = _pm_decision(
            rating=PortfolioRating.SELL,
            price_target_horizon=80.0,
            target_basis="dcf",
            pullback_zone=95.0,  # BELOW close → not a pullback for a short
            pullback_basis="moving_average",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_zone is None
        assert any(PULLBACK_DIRECTION_VIOLATION in n for n in result.notes)

    def test_hold_with_pullback_above_close_is_preserved(self):
        """Hold-with-upside-then-pullback is legitimate; Hold allows either side."""
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=103.0,
            target_basis="range_bound",
            pullback_zone=105.0,
            pullback_basis="prior_consolidation",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_zone == 105.0
        assert result.decision.pullback_basis == "prior_consolidation"


@pytest.mark.unit
class TestPullbackBasisVocabulary:
    """Slice 3: pullback_basis must come from the controlled vocabulary."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_unrecognised_pullback_basis_is_dropped(self):
        from tradingagents.agents.utils.decision_contracts import (
            PULLBACK_BASIS_UNRECOGNISED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="dcf",
            pullback_zone=92.0,
            pullback_basis="thesis_break",  # not in vocabulary
        )
        result = validate_portfolio_decision(decision, self._ctx())
        # Pullback zone is preserved; basis is dropped (vocab fail leaves
        # a defensible level surfaceable but strips the noisy label).
        # Note: orphan-basis cleanup will then sweep nothing because the
        # pullback_zone is still set after vocab clears the basis.
        assert result.decision.pullback_basis is None
        assert any(PULLBACK_BASIS_UNRECOGNISED in n for n in result.notes)

    def test_uppercase_vwap_anchor_is_coerced_to_canonical(self):
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=120.0,
            target_basis="dcf",
            pullback_zone=92.0,
            pullback_basis="VWAP_ANCHOR",  # uppercase
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_zone == 92.0
        assert result.decision.pullback_basis == "vwap_anchor"

    def test_orphan_pullback_basis_without_zone_is_cleared(self):
        """An orphan pullback_basis (no paired zone) is swept like target_basis."""
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            pullback_zone=None,
            pullback_basis="moving_average",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.pullback_basis is None


@pytest.mark.unit
class TestRenderPullbackAndDisagreementLines:
    """Slice 3: render_pm_decision must emit ``Pullback Zone`` and
    ``Disagreement Rationale`` lines when those fields are set."""

    def test_pullback_zone_renders_with_basis_label(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
            pullback_zone=92.0,
            pullback_basis="moving_average",
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Pullback Zone**: 92.0" in md
        assert "moving_average" in md

    def test_no_pullback_zone_means_no_pullback_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Pullback Zone**" not in md

    def test_disagreement_renders_as_named_line(self):
        from tradingagents.agents.schemas import (
            RatingTargetDisagreement,
            render_pm_decision,
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=97.0,
            rating_target_disagreement=RatingTargetDisagreement.QUALITY_PREMIUM,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Disagreement Rationale**: quality_premium" in md

    def test_disagreement_none_value_is_not_rendered(self):
        from tradingagents.agents.schemas import (
            RatingTargetDisagreement,
            render_pm_decision,
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
            rating_target_disagreement=RatingTargetDisagreement.NONE,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Disagreement Rationale**" not in md

    def test_no_disagreement_means_no_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=120.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        assert "**Disagreement Rationale**" not in md

    def test_pullback_and_disagreement_order_after_expected_return(self):
        from tradingagents.agents.schemas import (
            RatingTargetDisagreement,
            render_pm_decision,
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=103.0,
            pullback_zone=92.0,
            pullback_basis="moving_average",
            rating_target_disagreement=RatingTargetDisagreement.DIVIDEND_FLOOR,
            time_horizon="3-6 months",
        )
        md = render_pm_decision(decision, latest_close=100.0)
        # Order: Horizon Target / Expected Return / Pullback Zone /
        # Disagreement Rationale / Time Horizon.
        assert md.index("**Horizon Target**") < md.index("**Expected Return**")
        assert md.index("**Expected Return**") < md.index("**Pullback Zone**")
        assert md.index("**Pullback Zone**") < md.index("**Disagreement Rationale**")
        assert md.index("**Disagreement Rationale**") < md.index("**Time Horizon**")


@pytest.mark.unit
class TestDisagreementUsageMetric:
    """Slice 3: expose a tiny helper that aggregators can call to track
    rating_target_disagreement usage rate across runs. >5% is a smell."""

    def test_usage_rate_counts_non_none_disagreements(self):
        from tradingagents.agents.schemas import RatingTargetDisagreement
        from tradingagents.agents.utils.decision_contracts import (
            disagreement_usage_rate,
        )
        # 10 decisions, 1 with non-none disagreement → rate == 0.1.
        decisions = []
        for _ in range(9):
            decisions.append(
                ValidatedPortfolioDecision(
                    decision=_pm_decision(rating=PortfolioRating.HOLD),
                    notes=[],
                )
            )
        decisions.append(
            ValidatedPortfolioDecision(
                decision=_pm_decision(
                    rating=PortfolioRating.HOLD,
                    rating_target_disagreement=RatingTargetDisagreement.DIVIDEND_FLOOR,
                ),
                notes=[],
            )
        )
        assert disagreement_usage_rate(decisions) == 0.1

    def test_usage_rate_treats_none_value_as_no_disagreement(self):
        from tradingagents.agents.schemas import RatingTargetDisagreement
        from tradingagents.agents.utils.decision_contracts import (
            disagreement_usage_rate,
        )
        # Decisions whose disagreement field is the explicit NONE enum
        # value must not be counted as "in use".
        decisions = [
            ValidatedPortfolioDecision(
                decision=_pm_decision(
                    rating=PortfolioRating.HOLD,
                    rating_target_disagreement=RatingTargetDisagreement.NONE,
                ),
                notes=[],
            )
            for _ in range(5)
        ]
        assert disagreement_usage_rate(decisions) == 0.0

    def test_usage_rate_empty_list_returns_zero(self):
        from tradingagents.agents.utils.decision_contracts import (
            disagreement_usage_rate,
        )
        assert disagreement_usage_rate([]) == 0.0


@pytest.mark.unit
class TestPortfolioManagerSlice3Prompt:
    """Slice 3: the PM prompt must carry the ``Price Target Semantics``
    block so the model sees the controlled vocabularies and the explicit
    rules about ``pullback_zone`` vs ``price_target_horizon`` vs
    ``rating_target_disagreement``."""

    def _state(self, ticker="AAPL", latest_close=100.0):
        from tradingagents.agents.utils.evidence_ledger import (
            EvidenceFact,
            EvidenceLedger,
        )
        ledger = EvidenceLedger(
            ticker=ticker,
            trade_date="2026-05-11",
            latest_close=EvidenceFact(
                value=latest_close, source="yfinance", as_of="2026-05-11",
            ),
        )
        return {
            "company_of_interest": ticker,
            "trade_date": "2026-05-11",
            "past_context": "",
            "evidence_ledger": ledger,
            "risk_debate_state": {
                "history": "h",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "judge_decision": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "count": 1,
            },
            "investment_plan": "rp",
            "trader_investment_plan": "tp",
        }

    def _capturing_llm(self):
        from unittest.mock import MagicMock
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
        )
        captured = {}
        structured = MagicMock()

        def _capture(prompt):
            captured["prompt"] = prompt
            return decision

        structured.invoke.side_effect = _capture
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        return llm, captured

    def test_prompt_carries_price_target_semantics_block(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        llm, captured = self._capturing_llm()
        pm = create_portfolio_manager(llm)
        pm(self._state())
        prompt = captured["prompt"]
        assert "Price Target Semantics" in prompt

    def test_prompt_lists_target_basis_vocabulary(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        llm, captured = self._capturing_llm()
        pm = create_portfolio_manager(llm)
        pm(self._state())
        prompt = captured["prompt"]
        # Every canonical token must appear so the model can pick from it.
        for token in [
            "base_case", "bear_case_skew", "mean_reversion", "range_bound",
            "catalyst_neutral", "dcf", "peer_multiple", "peg_at_consensus",
            "technical_measured_move",
        ]:
            assert token in prompt, f"missing target_basis token: {token}"

    def test_prompt_lists_pullback_basis_vocabulary(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        llm, captured = self._capturing_llm()
        pm = create_portfolio_manager(llm)
        pm(self._state())
        prompt = captured["prompt"]
        for token in [
            "retest_breakout", "fibonacci", "prior_consolidation",
            "moving_average", "vwap_anchor", "support_zone",
        ]:
            assert token in prompt, f"missing pullback_basis token: {token}"

    def test_prompt_lists_disagreement_vocabulary(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        llm, captured = self._capturing_llm()
        pm = create_portfolio_manager(llm)
        pm(self._state())
        prompt = captured["prompt"]
        for token in [
            "dividend_floor", "quality_premium", "momentum_override",
            "structural_optionality", "merger_arb_floor",
        ]:
            assert token in prompt, f"missing disagreement token: {token}"


# ---------------------------------------------------------------------------
# Slice 4: range targets, ticker-class point reject, soft triangulation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTargetRangeFields:
    """Slice 4: PortfolioDecision gains optional ``target_range_low`` and
    ``target_range_high`` fields. A point target (``price_target_horizon``
    alone) is still legal; the range is the new canonical horizon form."""

    def test_range_fields_set_with_valid_bounds_instantiates(self):
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=60.0,
            target_range_low=55.0,
            target_range_high=70.0,
        )
        assert decision.target_range_low == 55.0
        assert decision.target_range_high == 70.0
        assert decision.price_target_horizon == 60.0

    def test_range_fields_absent_is_backwards_compatible(self):
        decision = _pm_decision(rating=PortfolioRating.HOLD)
        assert decision.target_range_low is None
        assert decision.target_range_high is None


@pytest.mark.unit
class TestTickerClassClassifier:
    """Slice 4: ``_ticker_class`` maps a yfinance-style ticker into one of
    ``{index, fx, macro_rate, commodity_future, equity}``. Used by the
    point-reject rule — index/FX/commodity tickers cannot carry a point
    target because the false-precision penalty is large at the index level."""

    def test_caret_prefix_is_index(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("^NSEI") == "index"
        assert _ticker_class("^GSPC") == "index"

    def test_eurusd_is_fx(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("EURUSD=X") == "fx"

    def test_known_macro_rate_symbols_are_macro_rate(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("^TNX") == "macro_rate"
        assert _ticker_class("^IRX") == "macro_rate"
        assert _ticker_class("DX-Y.NYB") == "macro_rate"

    def test_futures_suffix_is_commodity_future(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("CL=F") == "commodity_future"
        assert _ticker_class("GC=F") == "commodity_future"

    def test_indian_equity_is_equity(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("PARACABLES.NS") == "equity"

    def test_us_equity_is_equity(self):
        from tradingagents.agents.utils.decision_contracts import _ticker_class
        assert _ticker_class("AAPL") == "equity"


@pytest.mark.unit
class TestRangeValidator:
    """Slice 4: validator drops range bounds when they are structurally
    incoherent (one bound only, inverted, or inconsistent with the horizon
    median) and emits a named note."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_incomplete_range_drops_both_with_note(self):
        from tradingagents.agents.utils.decision_contracts import RANGE_INCOMPLETE
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
            target_range_low=105.0,
            # target_range_high missing
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.target_range_low is None
        assert result.decision.target_range_high is None
        assert any(RANGE_INCOMPLETE in n for n in result.notes)

    def test_inverted_range_drops_both_with_note(self):
        from tradingagents.agents.utils.decision_contracts import RANGE_INVERTED
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
            target_range_low=120.0,
            target_range_high=105.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.target_range_low is None
        assert result.decision.target_range_high is None
        assert any(RANGE_INVERTED in n for n in result.notes)

    def test_horizon_outside_range_drops_range_keeps_horizon(self):
        from tradingagents.agents.utils.decision_contracts import (
            RANGE_INCONSISTENT_WITH_HORIZON,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=130.0,  # outside [105, 120]
            target_basis="dcf",
            target_range_low=105.0,
            target_range_high=120.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        # Range is dropped, horizon target preserved.
        assert result.decision.target_range_low is None
        assert result.decision.target_range_high is None
        assert result.decision.price_target_horizon == 130.0
        assert any(RANGE_INCONSISTENT_WITH_HORIZON in n for n in result.notes)

    def test_valid_range_with_horizon_inside_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
            target_range_low=105.0,
            target_range_high=120.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.target_range_low == 105.0
        assert result.decision.target_range_high == 120.0
        assert result.decision.price_target_horizon == 110.0

    def test_buy_range_low_below_close_drops_range(self):
        """Range directional contract: a Buy with range_low BELOW close
        means the bottom of the range is a Sell signal — drop the range."""
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=115.0,
            target_basis="dcf",
            target_range_low=95.0,  # below close (100)
            target_range_high=120.0,
        )
        result = validate_portfolio_decision(decision, self._ctx())
        assert result.decision.target_range_low is None
        assert result.decision.target_range_high is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)


@pytest.mark.unit
class TestHorizonConditionalRangeRequired:
    """7ez — at swing+ horizons (swing / position / long-term) the PM must
    publish a target range; a bare point target hides path uncertainty over
    multi-week / multi-month windows. The validator emits
    TARGET_RANGE_REQUIRED and synthesizes a ±5% range around the point as a
    best-effort fallback so the renderer still produces a range row.
    """

    def _ctx_with_horizon(self, horizon: str) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
            horizon=horizon,
        )

    def test_swing_horizon_point_only_emits_required_note(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("swing"),
        )
        assert any(TARGET_RANGE_REQUIRED in n for n in result.notes)

    def test_swing_horizon_point_only_synthesizes_plus_minus_5pct_range(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_SYNTHESIZED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("swing"),
        )
        assert result.decision.target_range_low == pytest.approx(104.5)
        assert result.decision.target_range_high == pytest.approx(115.5)
        assert any(TARGET_RANGE_SYNTHESIZED in n for n in result.notes)

    def test_position_horizon_point_only_emits_required_note(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("position"),
        )
        assert any(TARGET_RANGE_REQUIRED in n for n in result.notes)

    def test_long_term_horizon_point_only_emits_required_note(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("long-term"),
        )
        assert any(TARGET_RANGE_REQUIRED in n for n in result.notes)

    def test_swing_horizon_with_explicit_range_emits_no_note(self):
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
            TARGET_RANGE_SYNTHESIZED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
            target_range_low=105.0,
            target_range_high=118.0,
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("swing"),
        )
        # Author-supplied range preserved verbatim; no synthesis note.
        assert result.decision.target_range_low == 105.0
        assert result.decision.target_range_high == 118.0
        assert not any(TARGET_RANGE_REQUIRED in n for n in result.notes)
        assert not any(TARGET_RANGE_SYNTHESIZED in n for n in result.notes)

    def test_no_horizon_set_is_silent_backward_compat(self):
        """Existing callers that don't thread horizon must keep working —
        the new rule activates only when horizon is one of the swing+
        labels."""
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )
        result = validate_portfolio_decision(decision, ctx)
        assert not any(TARGET_RANGE_REQUIRED in n for n in result.notes)

    def test_hold_rating_point_only_no_synthesis_at_swing(self):
        """A Hold-rated point target shouldn't trigger directional drops;
        the range is still synthesized because uncertainty applies to Holds
        too at swing+ horizons, but no TARGET_DIRECTION_VIOLATION should
        appear."""
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_DIRECTION_VIOLATION,
            TARGET_RANGE_SYNTHESIZED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=100.0,  # at-close, well inside Hold band
            target_basis="catalyst_neutral",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("swing"),
        )
        assert any(TARGET_RANGE_SYNTHESIZED in n for n in result.notes)
        assert not any(TARGET_DIRECTION_VIOLATION in n for n in result.notes)

    def test_disagreement_suspends_horizon_range_rule(self):
        """When ``rating_target_disagreement`` is set the PM has explicitly
        signalled an unusual posture — don't force a synthetic range that
        contradicts the rationale."""
        from tradingagents.agents.schemas import RatingTargetDisagreement
        from tradingagents.agents.utils.decision_contracts import (
            TARGET_RANGE_REQUIRED,
            TARGET_RANGE_SYNTHESIZED,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
            rating_target_disagreement=RatingTargetDisagreement.DIVIDEND_FLOOR,
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_horizon("swing"),
        )
        # When the directional contract is suspended, the horizon-range
        # rule is too — the PM owns the trade-off explicitly.
        assert not any(TARGET_RANGE_REQUIRED in n for n in result.notes)
        assert not any(TARGET_RANGE_SYNTHESIZED in n for n in result.notes)


@pytest.mark.unit
class TestTickerClassPointReject:
    """Slice 4: index, FX, macro-rate, and commodity-future tickers cannot
    carry a point target — the false-precision penalty is too large. The
    range form (target_range_low/high) is required for these classes."""

    def _ctx_with_class(self, ticker_class: str) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE,
            latest_close=100.0,
            close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
            ticker_class=ticker_class,
        )

    def test_index_with_point_target_only_is_rejected(self):
        from tradingagents.agents.utils.decision_contracts import (
            POINT_TARGET_INAPPROPRIATE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_class("index"),
        )
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert any(POINT_TARGET_INAPPROPRIATE in n for n in result.notes)

    def test_index_with_range_only_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            target_range_low=108.0,
            target_range_high=120.0,
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_class("index"),
        )
        assert result.decision.target_range_low == 108.0
        assert result.decision.target_range_high == 120.0

    def test_index_with_point_and_range_drops_point_keeps_range(self):
        from tradingagents.agents.utils.decision_contracts import (
            POINT_TARGET_INAPPROPRIATE,
        )
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=115.0,
            target_basis="dcf",
            target_range_low=108.0,
            target_range_high=120.0,
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_class("index"),
        )
        assert result.decision.price_target_horizon is None
        assert result.decision.target_basis is None
        assert result.decision.target_range_low == 108.0
        assert result.decision.target_range_high == 120.0
        assert any(POINT_TARGET_INAPPROPRIATE in n for n in result.notes)

    def test_equity_with_point_target_only_is_preserved(self):
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        result = validate_portfolio_decision(
            decision, self._ctx_with_class("equity"),
        )
        assert result.decision.price_target_horizon == 110.0
        assert result.decision.target_basis == "dcf"


@pytest.mark.unit
class TestRenderTargetRange:
    """Slice 4: renderer surfaces the range form as ``**Target Range**:
    <low> – <high>`` directly below the horizon target lines. Skipped when
    the range is trivial (low == high == horizon)."""

    def test_range_only_renders_target_range_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            target_range_low=44.0,
            target_range_high=75.0,
        )
        md = render_pm_decision(decision)
        assert "**Target Range**: 44.0 – 75.0" in md

    def test_range_with_horizon_renders_both_lines(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=58.0,
            target_range_low=44.0,
            target_range_high=72.0,
        )
        md = render_pm_decision(decision)
        assert "**Horizon Target**: 58.0" in md
        assert "**Target Range**: 44.0 – 72.0" in md
        # Order: Horizon Target / Target Range / Expected Return.
        assert md.index("**Horizon Target**") < md.index("**Target Range**")

    def test_trivial_range_does_not_render(self):
        """When low == high == horizon, the range adds nothing — suppress it."""
        from tradingagents.agents.schemas import render_pm_decision
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=58.0,
            target_range_low=58.0,
            target_range_high=58.0,
        )
        md = render_pm_decision(decision)
        assert "**Target Range**" not in md

    def test_no_range_no_target_range_line(self):
        from tradingagents.agents.schemas import render_pm_decision
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=58.0,
        )
        md = render_pm_decision(decision)
        assert "**Target Range**" not in md

    def test_range_only_expected_return_uses_midpoint(self):
        """When only the range is set (no horizon), Expected Return is
        computed from the midpoint of the range so the reader still gets
        a signed pct."""
        from tradingagents.agents.schemas import render_pm_decision
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            target_range_low=100.0,
            target_range_high=110.0,
        )
        md = render_pm_decision(decision, latest_close=100.0)
        # Midpoint is 105 → +5%.
        assert "**Expected Return**: +5.0%" in md


def _full_sc(**overrides):
    """Build a minimal EvidenceScorecard with explicit category scores."""
    from tradingagents.agents.schemas import Confidence, EvidenceScorecard
    base = dict(
        bull_case=0, bear_case=0, trend_technical=0,
        fundamental_quality=0, liquidity_risk=0, catalyst_clarity=0,
        macro_regime=0, valuation=0,
        confidence=Confidence.MEDIUM,
        bull_case_rationale="bull rationale placeholder",
        bear_case_rationale="bear rationale placeholder",
        trend_technical_rationale="trend rationale placeholder",
        fundamental_quality_rationale="fundamentals rationale placeholder",
        liquidity_risk_rationale="liquidity rationale placeholder",
        catalyst_clarity_rationale="catalyst rationale placeholder",
        macro_regime_rationale="macro rationale placeholder",
        valuation_rationale="valuation rationale placeholder",
        rating_rationale="r", invalidating_evidence="i",
    )
    base.update(overrides)
    return EvidenceScorecard(**base)


@pytest.mark.unit
class TestTriangulationValidator:
    """Slice 4: ``triangulate_portfolio_decision`` is the SOFT validator —
    it never drops fields, only emits advisory notes when the scorecard,
    rating, and target disagree about direction."""

    def _ctx(self) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,
        )

    def test_strong_bullish_scorecard_with_hold_rating_emits_divergence(self):
        from tradingagents.agents.utils.decision_contracts import (
            SCORECARD_RATING_DIVERGENCE,
            triangulate_portfolio_decision,
        )
        # Net = +1+1+1+1+0+0+1+0 = +5 → strongly bullish.
        sc = _full_sc(
            bull_case=1, bear_case=1, trend_technical=1,
            fundamental_quality=1, liquidity_risk=0, catalyst_clarity=0,
            macro_regime=1, valuation=0,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            scorecard=sc,
        )
        result = triangulate_portfolio_decision(decision, self._ctx())
        # Decision is not modified.
        assert result.decision is decision or result.decision == decision
        assert any(SCORECARD_RATING_DIVERGENCE in n for n in result.notes)

    def test_strong_bearish_scorecard_with_buy_rating_emits_divergence(self):
        from tradingagents.agents.utils.decision_contracts import (
            SCORECARD_RATING_DIVERGENCE,
            triangulate_portfolio_decision,
        )
        # Net = -1-1-1-1-1+0-1+1 = -5 → strongly bearish.
        sc = _full_sc(
            bull_case=-1, bear_case=-1, trend_technical=-1,
            fundamental_quality=-1, liquidity_risk=-1, catalyst_clarity=0,
            macro_regime=-1, valuation=1,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            scorecard=sc,
        )
        result = triangulate_portfolio_decision(decision, self._ctx())
        assert any(SCORECARD_RATING_DIVERGENCE in n for n in result.notes)

    def test_paracables_like_weak_net_with_hold_does_not_fire(self):
        """PARACABLES: scorecard net = +1-1+1-1-2+0+1-1 = -2, Rating=Hold.
        |net|=2 < 4 threshold → SOFT divergence must NOT fire."""
        from tradingagents.agents.utils.decision_contracts import (
            SCORECARD_RATING_DIVERGENCE,
            triangulate_portfolio_decision,
        )
        sc = _full_sc(
            bull_case=1, bear_case=-1, trend_technical=1,
            fundamental_quality=-1, liquidity_risk=-2, catalyst_clarity=0,
            macro_regime=1, valuation=-1,
        )
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            scorecard=sc,
        )
        result = triangulate_portfolio_decision(decision, self._ctx())
        assert not any(SCORECARD_RATING_DIVERGENCE in n for n in result.notes)

    def test_strong_bullish_scorecard_with_buy_rating_does_not_fire(self):
        """Consistent: rating matches scorecard, no divergence."""
        from tradingagents.agents.utils.decision_contracts import (
            SCORECARD_RATING_DIVERGENCE,
            triangulate_portfolio_decision,
        )
        sc = _full_sc(
            bull_case=1, bear_case=1, trend_technical=1,
            fundamental_quality=1, liquidity_risk=0, catalyst_clarity=0,
            macro_regime=1, valuation=0,
        )
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            scorecard=sc,
        )
        result = triangulate_portfolio_decision(decision, self._ctx())
        assert not any(SCORECARD_RATING_DIVERGENCE in n for n in result.notes)

    def test_no_scorecard_emits_no_notes(self):
        from tradingagents.agents.utils.decision_contracts import (
            triangulate_portfolio_decision,
        )
        decision = _pm_decision(rating=PortfolioRating.HOLD)
        result = triangulate_portfolio_decision(decision, self._ctx())
        assert result.notes == []


@pytest.mark.unit
class TestRenderTriangulationNotes:
    def test_empty_notes_render_empty_string(self):
        from tradingagents.agents.utils.decision_contracts import (
            render_triangulation_notes,
        )
        assert render_triangulation_notes([]) == ""

    def test_notes_render_with_advisory_header(self):
        from tradingagents.agents.utils.decision_contracts import (
            render_triangulation_notes,
        )
        md = render_triangulation_notes(["SCORECARD_RATING_DIVERGENCE: ..."])
        assert "**Triangulation Notes**" in md
        assert "advisory" in md.lower()
        assert "no fields modified" in md.lower()
        assert "- SCORECARD_RATING_DIVERGENCE" in md


@pytest.mark.unit
class TestPortfolioManagerSlice4Wiring:
    """Slice 4: end-to-end through ``create_portfolio_manager``.
    Confirms the ticker-class is plumbed into the validator and the
    soft triangulator is called after validation."""

    def _state(self, ticker="^NSEI", latest_close=100.0):
        from tradingagents.agents.utils.evidence_ledger import (
            EvidenceFact,
            EvidenceLedger,
        )
        ledger = EvidenceLedger(
            ticker=ticker,
            trade_date="2026-05-11",
            latest_close=EvidenceFact(
                value=latest_close, source="yfinance", as_of="2026-05-11",
            ),
        )
        return {
            "company_of_interest": ticker,
            "trade_date": "2026-05-11",
            "past_context": "",
            "evidence_ledger": ledger,
            "risk_debate_state": {
                "history": "h",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "judge_decision": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "count": 1,
            },
            "investment_plan": "rp",
            "trader_investment_plan": "tp",
        }

    def _llm_with_decision(self, decision: PortfolioDecision):
        from unittest.mock import MagicMock
        structured = MagicMock()
        structured.invoke.return_value = decision
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        return llm

    def test_index_ticker_with_point_target_emits_point_target_inappropriate(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=110.0,
            target_basis="dcf",
        )
        llm = self._llm_with_decision(decision)
        pm = create_portfolio_manager(llm)
        result = pm(self._state(ticker="^NSEI", latest_close=100.0))
        md = result["final_trade_decision"]
        assert "POINT_TARGET_INAPPROPRIATE" in md

    def test_strong_scorecard_hold_emits_triangulation_note(self):
        from tradingagents.agents.managers.portfolio_manager import (
            create_portfolio_manager,
        )
        sc = _full_sc(
            bull_case=1, bear_case=1, trend_technical=1,
            fundamental_quality=1, liquidity_risk=0, catalyst_clarity=0,
            macro_regime=1, valuation=0,
            tie_breaker="Balanced near-term; await catalyst.",
        )
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            scorecard=sc,
        )
        llm = self._llm_with_decision(decision)
        pm = create_portfolio_manager(llm)
        result = pm(self._state(ticker="AAPL", latest_close=100.0))
        md = result["final_trade_decision"]
        assert "**Triangulation Notes**" in md
        assert "SCORECARD_RATING_DIVERGENCE" in md


# ---------------------------------------------------------------------------
# Slice 5: Target-drift tracking across reports.
#
# A PM publishes a target at close 50, then re-runs two weeks later with
# close 58 (+16% drift) and silently carries the same target forward. The
# previous slices harden the contract WITHIN a single run; this slice
# tracks the target's vintage so the next run sees the drift and the PM
# is forced to either explicitly REAFFIRM or REVISE.
# ---------------------------------------------------------------------------


class TestSlice5SchemaFields:
    """The three new fields exist on PortfolioDecision and accept values."""

    def test_target_committed_at_field_present(self):
        d = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            target_committed_at="2026-05-11",
        )
        assert d.target_committed_at == "2026-05-11"

    def test_committed_close_field_present(self):
        d = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            committed_close=59.77,
        )
        assert d.committed_close == 59.77

    def test_target_drift_threshold_pct_default_is_15pct(self):
        d = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
        )
        assert d.target_drift_threshold_pct == 0.15


class TestSlice5RenderTargetVintage:
    """``render_pm_decision`` emits a Target Vintage line directly below the
    horizon target line when both vintage fields are populated."""

    def test_vintage_line_rendered_when_both_fields_present(self):
        from tradingagents.agents.schemas import render_pm_decision
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=60.0,
            target_basis="dcf",
            target_committed_at="2026-04-27",
            committed_close=50.0,
        )
        md = render_pm_decision(d)
        assert "**Target Vintage**: committed 2026-04-27 at 50.0" in md

    def test_vintage_line_appears_below_horizon_target(self):
        from tradingagents.agents.schemas import render_pm_decision
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=60.0,
            target_basis="dcf",
            target_committed_at="2026-04-27",
            committed_close=50.0,
        )
        md = render_pm_decision(d)
        assert md.index("**Horizon Target**") < md.index("**Target Vintage**")

    def test_vintage_line_absent_when_committed_close_missing(self):
        from tradingagents.agents.schemas import render_pm_decision
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=60.0,
            target_basis="dcf",
            target_committed_at="2026-04-27",
        )
        md = render_pm_decision(d)
        assert "**Target Vintage**" not in md

    def test_vintage_line_absent_when_target_committed_at_missing(self):
        from tradingagents.agents.schemas import render_pm_decision
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=60.0,
            target_basis="dcf",
            committed_close=50.0,
        )
        md = render_pm_decision(d)
        assert "**Target Vintage**" not in md


class TestValidateTargetDrift:
    """``validate_target_drift`` compares the current close against the
    prior decision's committed close and returns an LLM-facing directive
    string when drift exceeds threshold; ``None`` otherwise.

    Distinct from the existing hard validator: this returns a directive
    that the PM node injects into the PROMPT (not a Validation Notes
    footer), because the corrective action is the LLM's, not the
    validator's. The validator cannot decide whether to reaffirm or revise.
    """

    def test_returns_none_when_committed_close_is_none(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
        )
        assert validate_target_drift(prior, current_close=58.0) is None

    def test_returns_none_when_current_close_is_none(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
        )
        assert validate_target_drift(prior, current_close=None) is None

    def test_returns_none_when_threshold_is_none(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
            target_drift_threshold_pct=None,
        )
        assert validate_target_drift(prior, current_close=80.0) is None

    def test_returns_none_when_drift_below_threshold(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
        )
        # 4% drift, default threshold 15% → no directive.
        assert validate_target_drift(prior, current_close=52.0) is None

    def test_returns_directive_when_drift_exceeds_threshold(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
        )
        # 16% drift, default 15% threshold → directive.
        directive = validate_target_drift(prior, current_close=58.0)
        assert directive is not None
        assert isinstance(directive, str)
        assert "ATTENTION" in directive
        assert "50" in directive
        assert "58" in directive
        assert "2026-04-27" in directive
        # Drift pct rendered as signed pct with one decimal.
        assert "+16.0%" in directive
        # Mentions threshold and reaffirm/revise.
        assert "15%" in directive
        assert "REAFFIRM" in directive
        assert "REVISE" in directive

    def test_custom_threshold_triggers_at_lower_drift(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
            target_drift_threshold_pct=0.10,
        )
        # 12% drift with 10% threshold → directive (would NOT fire at 15%).
        directive = validate_target_drift(prior, current_close=56.0)
        assert directive is not None
        assert "10%" in directive

    def test_directive_uses_absolute_drift_for_downward_move(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_target_drift,
        )
        prior = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target_horizon=50.0,
            committed_close=50.0,
            target_committed_at="2026-04-27",
        )
        # -20% drift, default 15% threshold → directive.
        directive = validate_target_drift(prior, current_close=40.0)
        assert directive is not None
        assert "-20.0%" in directive


# ---------------------------------------------------------------------------
# Cross-agent coherence (Slice 6): Trader.action ↔ PortfolioDecision.rating.
#
# Regression source: SOUTHBANK.NS dual-run on 2026-05-12 — Run 1 published
# Action=Hold + Rating=Overweight + Price Target +43.9% above close. Each
# agent's structured output passed its own validator in isolation, but
# nothing checked the pair. The new validator catches strict-polarity
# incoherence (Buy/Sell across the Hold pivot) as a standalone note, and
# the band check inside ``validate_portfolio_decision`` narrows to the
# Hold band whenever the Trader said Hold — independent of PM rating.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateActionRatingCoherence:
    """``validate_action_rating_coherence`` flags strict-polarity disagreement
    between Trader.action and PortfolioDecision.rating. Hold-action paired
    with any rating is coherent (the Trader is waiting, the PM names the
    long-run posture); a Buy paired with Sell/Underweight (or the inverse)
    is not."""

    def test_buy_action_with_underweight_rating_is_flagged(self):
        from tradingagents.agents.utils.decision_contracts import (
            RATING_ACTION_INCOHERENT,
            validate_action_rating_coherence,
        )
        notes = validate_action_rating_coherence(
            TraderAction.BUY, PortfolioRating.UNDERWEIGHT,
        )
        assert any(RATING_ACTION_INCOHERENT in n for n in notes)

    def test_buy_action_with_sell_rating_is_flagged(self):
        from tradingagents.agents.utils.decision_contracts import (
            RATING_ACTION_INCOHERENT,
            validate_action_rating_coherence,
        )
        notes = validate_action_rating_coherence(
            TraderAction.BUY, PortfolioRating.SELL,
        )
        assert any(RATING_ACTION_INCOHERENT in n for n in notes)

    def test_sell_action_with_overweight_rating_is_flagged(self):
        from tradingagents.agents.utils.decision_contracts import (
            RATING_ACTION_INCOHERENT,
            validate_action_rating_coherence,
        )
        notes = validate_action_rating_coherence(
            TraderAction.SELL, PortfolioRating.OVERWEIGHT,
        )
        assert any(RATING_ACTION_INCOHERENT in n for n in notes)

    def test_sell_action_with_buy_rating_is_flagged(self):
        from tradingagents.agents.utils.decision_contracts import (
            RATING_ACTION_INCOHERENT,
            validate_action_rating_coherence,
        )
        notes = validate_action_rating_coherence(
            TraderAction.SELL, PortfolioRating.BUY,
        )
        assert any(RATING_ACTION_INCOHERENT in n for n in notes)

    def test_buy_action_with_buy_rating_is_coherent(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_action_rating_coherence,
        )
        assert validate_action_rating_coherence(
            TraderAction.BUY, PortfolioRating.BUY,
        ) == []

    def test_buy_action_with_overweight_rating_is_coherent(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_action_rating_coherence,
        )
        assert validate_action_rating_coherence(
            TraderAction.BUY, PortfolioRating.OVERWEIGHT,
        ) == []

    def test_sell_action_with_underweight_rating_is_coherent(self):
        from tradingagents.agents.utils.decision_contracts import (
            validate_action_rating_coherence,
        )
        assert validate_action_rating_coherence(
            TraderAction.SELL, PortfolioRating.UNDERWEIGHT,
        ) == []

    def test_hold_action_pairs_with_any_rating(self):
        """A Hold action means the Trader is waiting; the PM's wider rating
        is allowed to encode the long-run posture without the pair tripping
        the coherence rule. Strict-polarity violations come from Buy↔Sell-side
        crossings, not from a Hold + Overweight pair."""
        from tradingagents.agents.utils.decision_contracts import (
            validate_action_rating_coherence,
        )
        for r in (
            PortfolioRating.BUY, PortfolioRating.OVERWEIGHT,
            PortfolioRating.HOLD, PortfolioRating.UNDERWEIGHT,
            PortfolioRating.SELL,
        ):
            assert validate_action_rating_coherence(
                TraderAction.HOLD, r,
            ) == [], f"Hold + {r} should be coherent"

    def test_hold_rating_pairs_with_any_action(self):
        """Mirror of Hold-action — a Hold rating pairs with any trader action
        (the PM is neutral over the long run, the Trader takes the tactical
        position)."""
        from tradingagents.agents.utils.decision_contracts import (
            validate_action_rating_coherence,
        )
        for a in (TraderAction.BUY, TraderAction.HOLD, TraderAction.SELL):
            assert validate_action_rating_coherence(
                a, PortfolioRating.HOLD,
            ) == [], f"{a} + Hold should be coherent"


@pytest.mark.unit
class TestPortfolioContractTraderActionGatesBand:
    """``PortfolioValidationContext.trader_action``: when set, narrows the
    band check to whichever side is tighter — so a PM Overweight + Trader
    Hold + +30% target trips ``TARGET_DIRECTION_VIOLATION`` on the Hold
    band rather than passing on the wider Overweight band. This is the
    structural fix for the SOUTHBANK.NS Run 1 regression."""

    def _ctx(self, *, trader_action=None) -> PortfolioValidationContext:
        return PortfolioValidationContext(
            trade_date=_TRADE_DATE, latest_close=100.0, close_as_of=_TRADE_DATE,
            annualised_volatility=0.20,  # → 10% Hold band, same as Slice 1
            trader_action=trader_action,
        )

    def test_overweight_pm_with_hold_trader_outside_hold_band_is_dropped(self):
        """Run 1 regression: PM Overweight + Trader Hold + +30% target.
        Without trader-action gating the Overweight branch accepts +30% as
        bullish-coherent. With gating the Hold band applies and the target
        is dropped with TARGET_DIRECTION_VIOLATION."""
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=130.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.HOLD),
        )
        assert result.decision.price_target_horizon is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_overweight_pm_with_hold_trader_within_hold_band_passes(self):
        """The narrower band still admits targets within ±10%."""
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=105.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.HOLD),
        )
        assert result.decision.price_target_horizon == 105.0
        assert not any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_overweight_pm_range_with_hold_trader_outside_hold_band_is_dropped(self):
        """Run 1's at-a-glance also published Target Range 48–62 (a ~±15%
        spread around close). With trader-action gating, the wide-range
        bound trips TARGET_DIRECTION_VIOLATION on the Hold band."""
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            target_range_low=85.0,    # -15% (outside Hold band)
            target_range_high=130.0,  # +30% (outside Hold band)
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.HOLD),
        )
        assert result.decision.target_range_low is None
        assert result.decision.target_range_high is None
        assert any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_no_trader_action_falls_back_to_rating_band(self):
        """Backward compat: when ``trader_action`` is unset, the validator
        uses the PM rating's band exactly as before. Overweight + +5% passes,
        Overweight + +30% would also pass on the bullish branch (the
        directional epsilon is the only gate)."""
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=130.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(decision, self._ctx())
        # No gating → Overweight + +30% above close is fine on its own.
        assert result.decision.price_target_horizon == 130.0
        assert not any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)

    def test_hold_trader_with_hold_rating_unchanged(self):
        """Trader Hold + PM Hold + +5% target: nothing changes — both the
        rating-branch and the gated-Hold path agree on the Hold band."""
        decision = _pm_decision(
            rating=PortfolioRating.HOLD,
            price_target_horizon=105.0,
            target_basis="range_bound",
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.HOLD),
        )
        assert result.decision.price_target_horizon == 105.0
        assert result.notes == []

    def test_buy_trader_action_does_not_narrow_buy_band(self):
        """Buy trader + Buy rating + +5% target: no spurious tightening —
        the gating only kicks in when the Trader said Hold and the PM said
        a non-Hold rating. Buy/Buy stays on the bullish branch."""
        decision = _pm_decision(
            rating=PortfolioRating.BUY,
            price_target_horizon=130.0,
            target_basis="DCF",
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.BUY),
        )
        assert result.decision.price_target_horizon == 130.0
        assert result.notes == []

    def test_disagreement_set_suspends_gated_band(self):
        """The ``rating_target_disagreement`` escape hatch already suspends
        the directional contract; trader-action gating must not bypass it.
        A Hold-trader + Overweight-PM with +30% target AND a non-NONE
        disagreement value passes (informational note only)."""
        from tradingagents.agents.schemas import RatingTargetDisagreement
        decision = _pm_decision(
            rating=PortfolioRating.OVERWEIGHT,
            price_target_horizon=130.0,
            target_basis="DCF",
            rating_target_disagreement=RatingTargetDisagreement.MOMENTUM_OVERRIDE,
        )
        result = validate_portfolio_decision(
            decision, self._ctx(trader_action=TraderAction.HOLD),
        )
        assert result.decision.price_target_horizon == 130.0
        assert not any("TARGET_DIRECTION_VIOLATION" in n for n in result.notes)


@pytest.mark.unit
class TestPortfolioValidationContextTraderActionField:
    """Slice 6 extends the context with an optional Trader action handle."""

    def test_trader_action_defaults_to_none(self):
        ctx = PortfolioValidationContext(trade_date=_TRADE_DATE)
        assert ctx.trader_action is None

    def test_trader_action_accepts_explicit_value(self):
        ctx = PortfolioValidationContext(
            trade_date=_TRADE_DATE, trader_action=TraderAction.HOLD,
        )
        assert ctx.trader_action == TraderAction.HOLD


@pytest.mark.unit
class TestParseTraderAction:
    """``_parse_trader_action`` is the bridge between Trader output and the
    PM validator context. A silently-broken parser means the trader-action
    gate never kicks in — pin the contract."""

    def test_extracts_buy_from_final_transaction_proposal_line(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        text = "**Action**: Buy\n\nFINAL TRANSACTION PROPOSAL: **BUY**\n"
        assert _parse_trader_action(text) == TraderAction.BUY

    def test_extracts_hold(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        text = "FINAL TRANSACTION PROPOSAL: **HOLD**"
        assert _parse_trader_action(text) == TraderAction.HOLD

    def test_extracts_sell(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        text = "FINAL TRANSACTION PROPOSAL: **SELL**"
        assert _parse_trader_action(text) == TraderAction.SELL

    def test_case_insensitive(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        text = "final transaction proposal: **hold**"
        assert _parse_trader_action(text) == TraderAction.HOLD

    def test_no_match_returns_none(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        assert _parse_trader_action("I'd like to wait.") is None

    def test_empty_string_returns_none(self):
        from tradingagents.agents.managers.portfolio_manager import _parse_trader_action
        assert _parse_trader_action("") is None
        assert _parse_trader_action(None) is None  # type: ignore[arg-type]


@pytest.mark.unit
class TestPortfolioManagerSlice6Wiring(TestPortfolioManagerSlice2Wiring):
    """End-to-end wiring test: a Hold-trader-plan + Overweight-PM
    decision must trip TARGET_DIRECTION_VIOLATION on the Hold band even
    though the Overweight branch alone would accept the +30% target.
    This is the SOUTHBANK.NS Run 1 regression in fixture form."""

    def _state_with_trader_plan(
        self, *, ticker: str, latest_close: float, trader_plan: str,
        vol: Optional[float] = None,
    ):
        state = self._state(ticker=ticker, latest_close=latest_close, vol=vol)
        state["trader_investment_plan"] = trader_plan
        return state

    def test_hold_trader_plus_overweight_pm_drops_far_target(self):
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Bullish over the year, awaiting tactical entry.",
            investment_thesis="Setup constructive; near-term consolidation expected.",
            price_target_horizon=130.0,  # +30% — outside the Hold band
            target_basis="DCF",
        )
        pm = create_portfolio_manager(self._llm(decision))
        trader_plan = (
            "## Trader Proposal\n"
            "**Action**: Hold\n\n"
            "FINAL TRANSACTION PROPOSAL: **HOLD**\n"
        )
        result = pm(self._state_with_trader_plan(
            ticker="SOUTHBANK.NS",
            latest_close=100.0,
            vol=0.20,
            trader_plan=trader_plan,
        ))
        md = result["final_trade_decision"]
        assert "TARGET_DIRECTION_VIOLATION" in md, (
            "Hold-trader gate did not narrow the Overweight band — the "
            "+30% target slipped through (SOUTHBANK Run 1 regression)."
        )
        assert "**Horizon Target**: 130.0" not in md

    def test_buy_trader_plus_overweight_pm_preserves_far_target(self):
        """Mirror case: Buy-trader + Overweight-PM + +30% target → the
        Overweight band applies (bullish; any target above close + ε is
        coherent) and the target is preserved. Confirms the gate is
        Hold-specific and does NOT spuriously tighten Buy/Buy paths."""
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Accumulate now.",
            investment_thesis="Setup confirmed.",
            price_target_horizon=130.0,
            target_basis="DCF",
        )
        pm = create_portfolio_manager(self._llm(decision))
        trader_plan = (
            "## Trader Proposal\n"
            "**Action**: Buy\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**\n"
        )
        result = pm(self._state_with_trader_plan(
            ticker="SOUTHBANK.NS",
            latest_close=100.0,
            vol=0.20,
            trader_plan=trader_plan,
        ))
        md = result["final_trade_decision"]
        assert "**Horizon Target**: 130.0" in md
        assert "TARGET_DIRECTION_VIOLATION" not in md
