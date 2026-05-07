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
from typing import Optional

from tradingagents.agents.schemas import TraderAction, TraderProposal


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
