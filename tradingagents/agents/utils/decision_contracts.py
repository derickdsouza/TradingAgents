"""Decision-contract validators for typed agent output.

The structured-output schemas in :mod:`tradingagents.agents.schemas` keep
agent decisions parseable, but a parseable decision is not the same as a
*tradeable* decision. A long-side proposal can carry a ``stop_initial``
above the latest close — structurally valid for the schema, but
semantically a target/upgrade trigger, not a stop. Prompt text alone
cannot prevent this; the model occasionally still ships invariant
violations.

This module owns the deterministic post-parse checks. Callers run a
proposal through the relevant ``validate_*`` function before rendering
to markdown, and the renderer presents either a clean proposal or a
clean proposal plus a small ``Validation Notes`` footer naming the
fields that were normalised.

Scope notes:

- Only the Trader is validated in this slice. ``PortfolioDecision`` has
  no stop fields and thus no analogous structural check; revisit when
  the Manager Scorecard bead lands and the decision shape grows.
- The validator only *removes* fields that are provably invalid and
  records a note. It never invents a value, infers a corrected level,
  or rewrites the model's reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)


# ---------------------------------------------------------------------------
# Note-ID constants
#
# Slice 1 of the PM decision contract surfaces three machine-grep-able note
# IDs in validator output. Future slices (volatility-scaled bands, drift
# tracking, currency mismatch) extend this list; downstream evaluation
# harnesses match on these constants rather than free-text prose.
# ---------------------------------------------------------------------------

LATEST_CLOSE_STALE = "LATEST_CLOSE_STALE"
TARGET_DIRECTION_VIOLATION = "TARGET_DIRECTION_VIOLATION"
TARGET_ORPHAN_BASIS = "TARGET_ORPHAN_BASIS"


# Fixed Hold band for Slice 1. Slice 2 replaces this with a
# volatility-scaled band (typically 0.75 × ATR / close). The fixed band is
# wide enough to keep most legitimate Hold targets and narrow enough to
# catch the PARACABLES regression (-13% on a Hold).
_HOLD_BAND = 0.10

# Epsilon for "meaningfully above/below close" on directional ratings.
# 3% mirrors the value the bead description anchors on. Targets within
# epsilon of close on a Buy/Sell are statistical noise, not a target.
_DIRECTIONAL_EPSILON = 0.03

# Bullish / bearish polarity on the 5-tier scale. Hold is handled
# separately by the band check.
_BULLISH_RATINGS = (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT)
_BEARISH_RATINGS = (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT)


@dataclass(frozen=True)
class TraderValidationContext:
    """Inputs needed by the trader validator beyond the proposal itself.

    ``latest_close`` is the deterministic current price (typically parsed
    from the Key Price Levels block fed to the agent). When ``None`` the
    directional checks are skipped — without a reference price we cannot
    disprove a stop's direction.
    """

    latest_close: Optional[float] = None


@dataclass
class ValidatedTraderProposal:
    """Result of validating a TraderProposal.

    ``proposal`` is a copy of the input with any provably invalid fields
    cleared (the original input is not mutated). ``notes`` is a
    human-readable list of the adjustments made; an empty list means the
    proposal passed unchanged.
    """

    proposal: TraderProposal
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.notes)


def validate_trader_proposal(
    proposal: TraderProposal,
    context: TraderValidationContext,
) -> ValidatedTraderProposal:
    """Strip semantically invalid fields from a TraderProposal.

    Invariants enforced:

    1. **Directional stops vs latest close.** For long-side actions
       (Buy / Hold) ``stop_initial`` and ``stop_trailing`` must sit
       BELOW ``latest_close``; for short-side (Sell) they must sit ABOVE.
       A long stop above the close is a target/upgrade trigger, not a
       stop. Violators are dropped (value and basis cleared) with a note.

    2. **Long stop below entry.** When both ``entry_price`` and
       ``stop_initial`` are present on a long-side proposal, the stop
       must be below the entry. A stop above entry on a long is
       structurally a target. Violators are dropped with a note.

    3. **Orphan basis labels.** When a price field is ``None`` but its
       paired ``_basis`` label is populated, the orphan basis is
       cleared. The renderer skips the field anyway, but a stray basis
       can leak into other consumers (memory log, downstream parsers).

    The validator never invents prices or rewrites prose. When
    ``latest_close`` is None the directional check (1) is skipped; the
    entry-vs-stop check (2) and orphan-basis cleanup (3) still apply.
    """
    cleaned = proposal.model_copy()
    notes: list[str] = []
    close = context.latest_close

    is_long = proposal.action in (TraderAction.BUY, TraderAction.HOLD)
    is_short = proposal.action == TraderAction.SELL

    def _drop(value_attr: str, basis_attr: str, value: float, label: str, expected_side: str) -> None:
        setattr(cleaned, value_attr, None)
        setattr(cleaned, basis_attr, None)
        notes.append(
            f"{label} {value} dropped: must be {expected_side} latest close "
            f"{close} for action {proposal.action.value} — value was a target "
            f"or trigger, not a stop."
        )

    if close is not None:
        if is_long:
            if proposal.stop_initial is not None and proposal.stop_initial >= close:
                _drop("stop_initial", "stop_initial_basis", proposal.stop_initial, "Initial Stop", "below")
            if proposal.stop_trailing is not None and proposal.stop_trailing >= close:
                _drop("stop_trailing", "stop_trailing_basis", proposal.stop_trailing, "Trailing Stop", "below")
        elif is_short:
            if proposal.stop_initial is not None and proposal.stop_initial <= close:
                _drop("stop_initial", "stop_initial_basis", proposal.stop_initial, "Initial Stop", "above")
            if proposal.stop_trailing is not None and proposal.stop_trailing <= close:
                _drop("stop_trailing", "stop_trailing_basis", proposal.stop_trailing, "Trailing Stop", "above")

    if cleaned.entry_price is not None:
        if is_long and cleaned.stop_initial is not None and cleaned.stop_initial >= cleaned.entry_price:
            value = cleaned.stop_initial
            cleaned.stop_initial = None
            cleaned.stop_initial_basis = None
            notes.append(
                f"Initial Stop {value} dropped: must be below entry_price "
                f"{cleaned.entry_price} for a long-side action."
            )
        if is_short and cleaned.stop_initial is not None and cleaned.stop_initial <= cleaned.entry_price:
            value = cleaned.stop_initial
            cleaned.stop_initial = None
            cleaned.stop_initial_basis = None
            notes.append(
                f"Initial Stop {value} dropped: must be above entry_price "
                f"{cleaned.entry_price} for a short-side action."
            )

    for value_attr, basis_attr, label in [
        ("entry_price", "entry_basis", "Entry"),
        ("stop_initial", "stop_initial_basis", "Initial Stop"),
        ("stop_trailing", "stop_trailing_basis", "Trailing Stop"),
    ]:
        if getattr(cleaned, value_attr) is None and getattr(cleaned, basis_attr):
            setattr(cleaned, basis_attr, None)
            notes.append(f"{label} basis dropped: orphan basis without a paired price value.")

    return ValidatedTraderProposal(proposal=cleaned, notes=notes)


def render_validation_notes(notes: list[str]) -> str:
    """Render notes as a small markdown footer, or ``""`` when empty.

    The block is appended to the rendered proposal so the reader sees
    exactly which fields were normalised and why, without losing the
    cleaned headline numbers.
    """
    if not notes:
        return ""
    lines = ["**Validation Notes**:"]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Manager decision contract (Slice 1).
#
# Slice 1 of the PM decision contract enforces three rules on
# ``PortfolioDecision``:
#
#   1. **Loud-fail on missing/stale close.** Opposite of the Trader's
#      silent skip — a PM target without a reference price is unverifiable
#      and we refuse to publish unverifiable numbers.
#   2. **Directional coherence.** Buy/Overweight targets must sit
#      meaningfully above close; Sell/Underweight meaningfully below;
#      Hold within a fixed ±10% band. Slice 2 replaces the fixed band
#      with a volatility-scaled one.
#   3. **Orphan basis cleanup.** A ``target_basis`` label without a
#      paired ``price_target_horizon`` value is structurally a render bug.
#
# Slices 2-5 layer additional rules (currency mismatch, controlled-vocab
# bases, range targets, scorecard triangulation, drift tracking) on the
# same shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioValidationContext:
    """Inputs needed by the PM validator beyond the decision itself.

    ``trade_date`` is required so the staleness check has an anchor.
    ``latest_close`` and ``close_as_of`` come from the evidence ledger or
    the Key Price Levels block fed to the agent; when either is missing
    the validator drops the target loudly (LATEST_CLOSE_STALE) rather
    than skipping the check, because an unverifiable target is worse
    than no target at all.
    """

    trade_date: datetime
    latest_close: Optional[float] = None
    close_as_of: Optional[datetime] = None


@dataclass
class ValidatedPortfolioDecision:
    """Result of validating a PortfolioDecision.

    ``decision`` is a copy of the input with any provably invalid fields
    cleared (the original input is not mutated). ``notes`` is a
    human-readable list of the adjustments made; an empty list means the
    decision passed unchanged.
    """

    decision: PortfolioDecision
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.notes)


def _close_is_stale(ctx: PortfolioValidationContext) -> bool:
    """Whether the reference close is missing or older than ~1 session.

    "1 session" is approximated as 1 calendar day for Slice 1; Slice 2
    refines this with an exchange-calendar lookup. The check is
    deliberately conservative — better to drop a target on a borderline
    fresh close than to publish one anchored to a stale price.
    """
    if ctx.latest_close is None or ctx.close_as_of is None:
        return True
    delta = ctx.trade_date - ctx.close_as_of
    return delta > timedelta(days=1)


def validate_portfolio_decision(
    decision: PortfolioDecision,
    context: PortfolioValidationContext,
) -> ValidatedPortfolioDecision:
    """Strip semantically invalid fields from a PortfolioDecision.

    Invariants enforced (Slice 1):

    1. **Loud-fail on missing or stale close.** If ``latest_close`` is
       absent or ``close_as_of`` is more than ~1 session before
       ``trade_date``, ``price_target_horizon`` and ``target_basis`` are
       dropped with a ``LATEST_CLOSE_STALE`` note. This is the OPPOSITE
       of the Trader's silent skip — a PM target without a reference
       price is unverifiable and we refuse to publish unverifiable
       numbers in the headline.

    2. **Directional coherence.** For Buy/Overweight, the target must
       be more than ``_DIRECTIONAL_EPSILON`` above close. For
       Sell/Underweight, more than ``_DIRECTIONAL_EPSILON`` below. For
       Hold, within ``_HOLD_BAND`` of close (fixed Slice-1
       placeholder; Slice 2 replaces with volatility-scaled).
       Violators drop both value and basis with a
       ``TARGET_DIRECTION_VIOLATION`` note.

    3. **Orphan basis cleanup.** If ``target_basis`` is set but
       ``price_target_horizon`` is None (either originally or after
       direction-cleanup), the orphan basis is cleared with a
       ``TARGET_ORPHAN_BASIS`` note.

    The validator never invents prices and never rewrites prose.
    """
    cleaned = decision.model_copy()
    notes: list[str] = []

    # Rule 1: loud-fail when the reference close is missing or stale.
    if cleaned.price_target_horizon is not None and _close_is_stale(context):
        original = cleaned.price_target_horizon
        cleaned.price_target_horizon = None
        cleaned.target_basis = None
        notes.append(
            f"{LATEST_CLOSE_STALE}: Price Target {original} dropped — "
            f"latest_close is missing or older than the trade date, so "
            f"the target's direction cannot be verified."
        )

    # Rule 2: directional contract against latest_close.
    if (
        cleaned.price_target_horizon is not None
        and context.latest_close is not None
    ):
        close = context.latest_close
        target = cleaned.price_target_horizon
        rating = cleaned.rating
        violation: Optional[str] = None

        if rating in _BULLISH_RATINGS:
            min_target = close * (1.0 + _DIRECTIONAL_EPSILON)
            if target <= min_target:
                violation = (
                    f"target {target} is not meaningfully above close "
                    f"{close} for a {rating.value} rating (requires "
                    f">+{_DIRECTIONAL_EPSILON:.0%})."
                )
        elif rating in _BEARISH_RATINGS:
            max_target = close * (1.0 - _DIRECTIONAL_EPSILON)
            if target >= max_target:
                violation = (
                    f"target {target} is not meaningfully below close "
                    f"{close} for a {rating.value} rating (requires "
                    f"<-{_DIRECTIONAL_EPSILON:.0%})."
                )
        elif rating == PortfolioRating.HOLD:
            upper = close * (1.0 + _HOLD_BAND)
            lower = close * (1.0 - _HOLD_BAND)
            if target > upper or target < lower:
                violation = (
                    f"target {target} sits outside the ±{_HOLD_BAND:.0%} "
                    f"Hold band around close {close} — a target this far "
                    f"from close is a directional call, not a Hold."
                )

        if violation is not None:
            cleaned.price_target_horizon = None
            cleaned.target_basis = None
            notes.append(f"{TARGET_DIRECTION_VIOLATION}: {violation}")

    # Rule 3: orphan basis cleanup.
    if cleaned.price_target_horizon is None and cleaned.target_basis:
        cleaned.target_basis = None
        notes.append(
            f"{TARGET_ORPHAN_BASIS}: target_basis dropped — orphan basis "
            f"without a paired price_target_horizon value."
        )

    return ValidatedPortfolioDecision(decision=cleaned, notes=notes)


def render_pm_validation_notes(notes: list[str]) -> str:
    """Render PM-validator notes as a small markdown footer, or ``""``.

    Same shape as ``render_validation_notes`` for the Trader; kept as a
    named alias so callers in ``portfolio_manager.py`` read clearly and
    so the two footers can diverge later (e.g. PM-specific severity
    headers) without churn at the call sites.
    """
    return render_validation_notes(notes)
