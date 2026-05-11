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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
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
            target_basis="range midpoint",
        )
        pm = create_portfolio_manager(self._llm(decision))
        result = pm(self._state(ticker="AAPL", latest_close=100.0, vol=None))
        md = result["final_trade_decision"]
        # Target inside ±10% so preserved, but informational note fires.
        assert "**Horizon Target**: 103.0" in md
        assert "HOLD_BAND_UNCALIBRATED" in md
