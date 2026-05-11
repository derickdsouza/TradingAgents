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

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    RatingTargetDisagreement,
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

# Slice 2 additions.
HOLD_BAND_UNCALIBRATED = "HOLD_BAND_UNCALIBRATED"
TARGET_EQUALS_CLOSE = "TARGET_EQUALS_CLOSE"
TARGET_CURRENCY_MISMATCH = "TARGET_CURRENCY_MISMATCH"

# Slice 3 additions.
RATING_TARGET_DISAGREEMENT_SET = "RATING_TARGET_DISAGREEMENT_SET"
TARGET_BASIS_UNRECOGNISED = "TARGET_BASIS_UNRECOGNISED"
PULLBACK_BASIS_UNRECOGNISED = "PULLBACK_BASIS_UNRECOGNISED"
PULLBACK_DIRECTION_VIOLATION = "PULLBACK_DIRECTION_VIOLATION"

# Slice 4 additions.
RANGE_INCOMPLETE = "RANGE_INCOMPLETE"
RANGE_INVERTED = "RANGE_INVERTED"
RANGE_INCONSISTENT_WITH_HORIZON = "RANGE_INCONSISTENT_WITH_HORIZON"
POINT_TARGET_INAPPROPRIATE = "POINT_TARGET_INAPPROPRIATE"
SCORECARD_RATING_DIVERGENCE = "SCORECARD_RATING_DIVERGENCE"
SCORECARD_TARGET_DIVERGENCE = "SCORECARD_TARGET_DIVERGENCE"

# Slice 3 controlled vocabularies. Enforced post-parse so the schema stays
# simple (free-text fields) and vocabulary failures surface as named notes
# alongside the rest of the structural checks.
ALLOWED_TARGET_BASIS: frozenset[str] = frozenset({
    "base_case",
    "bear_case_skew",
    "mean_reversion",
    "range_bound",
    "catalyst_neutral",
    "dcf",
    "peer_multiple",
    "peg_at_consensus",
    "technical_measured_move",
})

ALLOWED_PULLBACK_BASIS: frozenset[str] = frozenset({
    "retest_breakout",
    "fibonacci",
    "prior_consolidation",
    "moving_average",
    "vwap_anchor",
    "support_zone",
})


# ---------------------------------------------------------------------------
# Ticker-suffix → ISO-4217 currency derivation.
#
# yfinance tickers carry a regional suffix (``.NS`` for NSE, ``.L`` for
# LSE, etc.). The PM contract defaults ``close_currency`` from the suffix
# so callers don't have to plumb explicit currencies for every regional
# exchange. Unknown / no suffix falls back to USD because that is the
# yfinance default for unsuffixed US tickers.
# ---------------------------------------------------------------------------

_SUFFIX_TO_CURRENCY: dict[str, str] = {
    "NS": "INR",
    "BO": "INR",
    "L": "GBP",
    "HK": "HKD",
    "TO": "CAD",
    "AX": "AUD",
    "T": "JPY",
    "JP": "JPY",
    "PA": "EUR",
    "AS": "EUR",
    "DE": "EUR",
    "MI": "EUR",
    "MC": "EUR",
    "BR": "EUR",
}


def _currency_from_ticker(ticker: str) -> Optional[str]:
    """Map a yfinance-style ticker to its ISO-4217 quote currency.

    Returns ``USD`` for tickers without a recognised suffix (the yfinance
    default for unsuffixed US tickers). Returns ``None`` only when the
    input is falsy.
    """
    if not ticker:
        return None
    if "." not in ticker:
        return "USD"
    suffix = ticker.rsplit(".", 1)[-1].upper()
    return _SUFFIX_TO_CURRENCY.get(suffix, "USD")


# ---------------------------------------------------------------------------
# Ticker-class classifier (Slice 4).
#
# Indices, FX pairs, macro rate series, and commodity futures cannot carry
# a defensible point target at a 12-month horizon — the false-precision
# penalty is large at the index level. The classifier groups tickers into
# the coarse buckets the point-reject rule branches on; equities are the
# only class permitted to use a point target.
# ---------------------------------------------------------------------------

# Specific yfinance symbols that are interest-rate / DXY series rather
# than indexes proper. Kept as an explicit set so a typo in a feed doesn't
# silently get reclassified as a regular index.
_MACRO_RATE_SYMBOLS: frozenset[str] = frozenset({
    "^TNX",  # 10-year Treasury yield
    "^IRX",  # 13-week Treasury bill yield
    "^FVX",  # 5-year Treasury yield
    "^TYX",  # 30-year Treasury yield
    "DX-Y.NYB",  # US dollar index
})


def _ticker_class(ticker: str) -> str:
    """Classify a yfinance-style ticker into a coarse instrument class.

    Buckets:

    - ``"index"``       — broad-market indexes (``^NSEI``, ``^GSPC``).
    - ``"fx"``          — currency pairs (``EURUSD=X``, ``JPY=X``).
    - ``"macro_rate"``  — interest-rate / DXY series (``^TNX``, ``DX-Y.NYB``).
    - ``"commodity_future"`` — futures contracts (``CL=F``, ``GC=F``).
    - ``"equity"``      — everything else (the default).

    The classifier is intentionally coarse: it just needs to know whether
    a point target is defensible on this instrument. Refining the rules
    (sector indexes, single-stock futures, etc.) is a follow-up; for now
    if a real edge case shows up the test fixture catches it and the
    classifier grows a branch.
    """
    if not ticker:
        return "equity"
    upper = ticker.upper()
    # Macro-rate symbols come first so they aren't swallowed by the
    # generic ``^`` index branch.
    if upper in _MACRO_RATE_SYMBOLS:
        return "macro_rate"
    if upper.startswith("^"):
        return "index"
    if upper.endswith("=F"):
        return "commodity_future"
    if upper.endswith("=X"):
        return "fx"
    return "equity"


# Fallback Hold band when volatility is unavailable. Slice 1 used this as
# a fixed band; Slice 2 keeps it as the fallback inside
# ``PortfolioValidationContext.hold_band_pct_default`` and also as the
# anchor when callers do not override the context default. Wide enough to
# keep most legitimate Hold targets and narrow enough to catch the
# PARACABLES regression (-13% on a Hold).
_HOLD_BAND_DEFAULT = 0.10

# Volatility-scaled band envelope: at least 5% (calm names get a sane
# minimum), at most 20% (the band stops widening for tail-risk names so a
# Hold doesn't become an unbounded target).
_HOLD_BAND_MIN = 0.05
_HOLD_BAND_MAX = 0.20

# Horizon used when scaling the Hold band by volatility. Slice 2 fixes
# this at 1.0 year; Slice 4+ will parse ``time_horizon`` text. A 1-year
# horizon is the natural pairing with annualised vol — sqrt(horizon)=1.0
# and the formula reduces to ``0.5 * annualised_vol``.
_HOLD_BAND_HORIZON_YEARS_DEFAULT = 1.0

# Placeholder-equality tolerance for the banned-placeholder rule.
# Targets within 0.5% of close are noise / placeholders, not a defensible
# horizon level. Slice 3 will add a ``catalyst_neutral`` opt-out via a
# controlled-vocab ``target_basis``; Slice 2 has no opt-out (the field is
# still free-text).
_TARGET_PLACEHOLDER_EPSILON = 0.005

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

    Slice 2 additions:

    - ``annualised_volatility``: annualised stdev of daily returns (or
      ``ATR / close * sqrt(252)``). Used to scale the Hold band. When
      ``None`` the validator falls back to ``hold_band_pct_default`` and
      emits ``HOLD_BAND_UNCALIBRATED``.
    - ``close_currency``: ISO-4217 currency of ``latest_close`` (typically
      derived from the ticker suffix by callers). Used by the currency-
      mismatch rule.
    - ``hold_band_pct_default``: fallback Hold band when volatility is
      unavailable. 10% mirrors Slice 1's fixed band.
    """

    trade_date: datetime
    latest_close: Optional[float] = None
    close_as_of: Optional[datetime] = None
    annualised_volatility: Optional[float] = None
    close_currency: Optional[str] = None
    hold_band_pct_default: float = 0.10
    # Slice 4: coarse instrument-class label so the validator can reject
    # point targets on instruments that have no defensible point view at
    # a 12-month horizon (indexes, FX, macro rates, commodity futures).
    # Callers populate from ``_ticker_class(ticker)``.
    ticker_class: Optional[str] = None


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

    Invariants enforced (Slice 1 + Slice 2). Rules run in this order so
    later rules see the cleaned-up state of earlier ones:

    1. **Loud-fail on missing or stale close** (Slice 1). If
       ``latest_close`` is absent or ``close_as_of`` is more than ~1
       session before ``trade_date``, ``price_target_horizon`` and
       ``target_basis`` are dropped with ``LATEST_CLOSE_STALE``. This is
       the OPPOSITE of the Trader's silent skip — a PM target without a
       reference price is unverifiable and we refuse to publish
       unverifiable numbers in the headline.

    2. **Currency mismatch** (Slice 2). When both ``target_currency``
       and ``close_currency`` are known and disagree, the target is
       dropped with ``TARGET_CURRENCY_MISMATCH``. A $120 target on an
       INR stock is a translation bug, not a horizon level.

    3. **Banned placeholder** (Slice 2). When the target sits within
       ``_TARGET_PLACEHOLDER_EPSILON`` (0.5%) of close, it is dropped
       with ``TARGET_EQUALS_CLOSE``. Slice 3 will add a
       ``catalyst_neutral`` opt-out via the controlled-vocab
       ``target_basis``; Slice 2 has no opt-out (the field is still
       free-text).

    4. **Directional coherence** (Slice 1 + Slice 2). For
       Buy/Overweight, the target must be more than
       ``_DIRECTIONAL_EPSILON`` above close; for Sell/Underweight, more
       than ``_DIRECTIONAL_EPSILON`` below. For Hold, the band scales
       with ``annualised_volatility`` and is clamped to
       ``[_HOLD_BAND_MIN, _HOLD_BAND_MAX]``; falls back to
       ``context.hold_band_pct_default`` with ``HOLD_BAND_UNCALIBRATED``
       when volatility is unavailable. Violators drop value and basis
       with ``TARGET_DIRECTION_VIOLATION``.

    5. **Orphan basis cleanup** (Slice 1). If ``target_basis`` is set
       but ``price_target_horizon`` is None (originally or after any of
       the drops above), the basis is cleared with
       ``TARGET_ORPHAN_BASIS``.

    The validator never invents prices and never rewrites prose.
    """
    cleaned = decision.model_copy()
    notes: list[str] = []

    # Slice 3: a non-none ``rating_target_disagreement`` is the explicit
    # escape hatch for legitimate contrarian positions. When set, the
    # banned-placeholder (rule 3) and directional contract (rule 4) checks
    # are SKIPPED for this decision; an informational note is emitted so
    # the rendered footer surfaces the rationale.
    disagreement = cleaned.rating_target_disagreement
    contract_suspended = (
        disagreement is not None
        and disagreement != RatingTargetDisagreement.NONE
    )
    if contract_suspended:
        notes.append(
            f"{RATING_TARGET_DISAGREEMENT_SET}: directional contract "
            f"suspended — rating_target_disagreement={disagreement.value}."
        )

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

    # Rule 2 (Slice 2): currency mismatch. When both target_currency and
    # close_currency are known and disagree, the target is meaningless
    # (a USD target on an INR-denominated stock is a translation bug, not
    # a horizon level). Runs early so the orphan-basis cleanup sweeps the
    # paired basis at the end.
    if (
        cleaned.price_target_horizon is not None
        and cleaned.target_currency is not None
        and context.close_currency is not None
        and cleaned.target_currency != context.close_currency
    ):
        original = cleaned.price_target_horizon
        cleaned.price_target_horizon = None
        cleaned.target_basis = None
        notes.append(
            f"{TARGET_CURRENCY_MISMATCH}: Price Target {original} "
            f"({cleaned.target_currency}) dropped — close is denominated "
            f"in {context.close_currency}, so the target cannot be "
            f"interpreted as a level in the instrument's quote currency."
        )

    # Rule 3 (Slice 2): banned placeholder — target within 0.5% of close
    # is statistical noise / a placeholder, not a defensible horizon
    # level. Slice 3 adds two opt-outs: (a) ``target_basis ==
    # 'catalyst_neutral'`` for explicit binary-event positions where
    # target ≈ close is intentional, and (b) any non-none
    # ``rating_target_disagreement`` (the contract is already suspended).
    target_basis_lc = (cleaned.target_basis or "").strip().lower()
    catalyst_neutral_opt_out = target_basis_lc == "catalyst_neutral"
    if (
        cleaned.price_target_horizon is not None
        and context.latest_close is not None
        and context.latest_close > 0
        and not contract_suspended
        and not catalyst_neutral_opt_out
    ):
        rel = abs(cleaned.price_target_horizon - context.latest_close) / context.latest_close
        if rel < _TARGET_PLACEHOLDER_EPSILON:
            original = cleaned.price_target_horizon
            cleaned.price_target_horizon = None
            cleaned.target_basis = None
            notes.append(
                f"{TARGET_EQUALS_CLOSE}: Price Target {original} dropped — "
                f"target sits within {_TARGET_PLACEHOLDER_EPSILON:.1%} of "
                f"close {context.latest_close}, which is a placeholder, "
                f"not a defensible horizon target."
            )

    # Rule 4: directional contract against latest_close. Slice 3:
    # SUSPENDED when ``rating_target_disagreement`` is set to a non-none
    # value — the PM has explicitly named a reason the target may
    # contradict the rating direction.
    if (
        cleaned.price_target_horizon is not None
        and context.latest_close is not None
        and not contract_suspended
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
            # Volatility-scaled band (Slice 2). Falls back to the
            # context default with a HOLD_BAND_UNCALIBRATED note when
            # annualised_volatility is unavailable.
            if context.annualised_volatility is not None:
                hold_band_pct = max(
                    _HOLD_BAND_MIN,
                    min(
                        _HOLD_BAND_MAX,
                        0.5
                        * context.annualised_volatility
                        * math.sqrt(_HOLD_BAND_HORIZON_YEARS_DEFAULT),
                    ),
                )
            else:
                hold_band_pct = context.hold_band_pct_default
                notes.append(
                    f"{HOLD_BAND_UNCALIBRATED}: annualised_volatility "
                    f"missing — Hold band defaulted to "
                    f"±{hold_band_pct:.0%} rather than a volatility-"
                    f"scaled value."
                )
            upper = close * (1.0 + hold_band_pct)
            lower = close * (1.0 - hold_band_pct)
            if target > upper or target < lower:
                violation = (
                    f"target {target} sits outside the ±{hold_band_pct:.0%} "
                    f"Hold band around close {close} — a target this far "
                    f"from close is a directional call, not a Hold."
                )

        if violation is not None:
            cleaned.price_target_horizon = None
            cleaned.target_basis = None
            notes.append(f"{TARGET_DIRECTION_VIOLATION}: {violation}")

    # Rule 4b (Slice 4): directional contract for range bounds. The range
    # bounds must point the same direction as the rating; for Buy/Overweight
    # BOTH bounds must sit above ``close * (1 + epsilon)``, for
    # Sell/Underweight BOTH below ``close * (1 - epsilon)``, for Hold BOTH
    # within the vol-scaled band. A violation drops BOTH bounds together
    # (a half-valid range is structurally a point, not a range). Suspended
    # under ``rating_target_disagreement`` just like the point variant.
    if (
        cleaned.target_range_low is not None
        and cleaned.target_range_high is not None
        and context.latest_close is not None
        and not contract_suspended
    ):
        close = context.latest_close
        low = cleaned.target_range_low
        high = cleaned.target_range_high
        rating = cleaned.rating
        range_violation: Optional[str] = None

        if rating in _BULLISH_RATINGS:
            min_target = close * (1.0 + _DIRECTIONAL_EPSILON)
            if low <= min_target or high <= min_target:
                range_violation = (
                    f"range [{low}, {high}] has a bound not meaningfully "
                    f"above close {close} for a {rating.value} rating "
                    f"(requires both bounds >+{_DIRECTIONAL_EPSILON:.0%})."
                )
        elif rating in _BEARISH_RATINGS:
            max_target = close * (1.0 - _DIRECTIONAL_EPSILON)
            if low >= max_target or high >= max_target:
                range_violation = (
                    f"range [{low}, {high}] has a bound not meaningfully "
                    f"below close {close} for a {rating.value} rating "
                    f"(requires both bounds <-{_DIRECTIONAL_EPSILON:.0%})."
                )
        elif rating == PortfolioRating.HOLD:
            # Mirror the point-target Hold band computation. Re-derive
            # here rather than thread state through the prior branch.
            if context.annualised_volatility is not None:
                hold_band_pct = max(
                    _HOLD_BAND_MIN,
                    min(
                        _HOLD_BAND_MAX,
                        0.5
                        * context.annualised_volatility
                        * math.sqrt(_HOLD_BAND_HORIZON_YEARS_DEFAULT),
                    ),
                )
            else:
                hold_band_pct = context.hold_band_pct_default
            upper = close * (1.0 + hold_band_pct)
            lower = close * (1.0 - hold_band_pct)
            if low < lower or high > upper:
                range_violation = (
                    f"range [{low}, {high}] has a bound outside the "
                    f"±{hold_band_pct:.0%} Hold band around close {close} "
                    f"— a range this wide is a directional call, not a Hold."
                )

        if range_violation is not None:
            cleaned.target_range_low = None
            cleaned.target_range_high = None
            notes.append(f"{TARGET_DIRECTION_VIOLATION}: {range_violation}")

    # Rule 4c (Slice 4): range coherence — incomplete, inverted, or
    # inconsistent-with-horizon. Runs after the directional contract so a
    # directional drop above doesn't trigger a spurious incomplete note.
    low = cleaned.target_range_low
    high = cleaned.target_range_high
    if (low is None) ^ (high is None):
        which = "high" if low is not None else "low"
        cleaned.target_range_low = None
        cleaned.target_range_high = None
        notes.append(
            f"{RANGE_INCOMPLETE}: target_range_{which} missing — both "
            f"bounds must be set together; dropping the orphan bound."
        )
    elif low is not None and high is not None:
        if low > high:
            cleaned.target_range_low = None
            cleaned.target_range_high = None
            notes.append(
                f"{RANGE_INVERTED}: target_range_low {low} > "
                f"target_range_high {high} — bounds dropped."
            )
        elif (
            cleaned.price_target_horizon is not None
            and not (low <= cleaned.price_target_horizon <= high)
        ):
            horizon = cleaned.price_target_horizon
            cleaned.target_range_low = None
            cleaned.target_range_high = None
            notes.append(
                f"{RANGE_INCONSISTENT_WITH_HORIZON}: price_target_horizon "
                f"{horizon} sits outside range [{low}, {high}] — range "
                f"dropped (horizon target preserved as authoritative)."
            )

    # Rule 4d (Slice 4): ticker-class point reject. Indexes, FX pairs,
    # macro-rate series, and commodity futures have no defensible point
    # target at a 12-month horizon — the false-precision penalty is too
    # large. When the instrument class is one of these and a point target
    # is present (with no range), drop the point and direct the PM to the
    # range form. Suspended under ``rating_target_disagreement`` so a
    # named escape hatch can still override.
    _POINT_INAPPROPRIATE_CLASSES = (
        "index", "fx", "macro_rate", "commodity_future",
    )
    if (
        context.ticker_class in _POINT_INAPPROPRIATE_CLASSES
        and cleaned.price_target_horizon is not None
        and not contract_suspended
    ):
        original = cleaned.price_target_horizon
        cleaned.price_target_horizon = None
        cleaned.target_basis = None
        notes.append(
            f"{POINT_TARGET_INAPPROPRIATE}: Price Target {original} "
            f"dropped — ticker_class '{context.ticker_class}' requires "
            f"the range form (target_range_low/target_range_high); point "
            f"targets at 12-month horizons are false precision on this "
            f"instrument class."
        )

    # Rule 5 (Slice 3): basis vocabulary validation. Coerce recognised
    # values to canonical lowercase; drop unrecognised values with a
    # named note. The target itself is preserved when only the basis is
    # bad — a defensible target with a noisy basis is still surfaceable.
    if cleaned.target_basis is not None:
        canonical, ok = _normalise_basis(cleaned.target_basis, ALLOWED_TARGET_BASIS)
        if ok:
            cleaned.target_basis = canonical
        else:
            original = cleaned.target_basis
            cleaned.target_basis = None
            notes.append(
                f"{TARGET_BASIS_UNRECOGNISED}: target_basis '{original}' "
                f"dropped — not in the controlled vocabulary "
                f"({sorted(ALLOWED_TARGET_BASIS)})."
            )

    if cleaned.pullback_basis is not None:
        canonical, ok = _normalise_basis(
            cleaned.pullback_basis, ALLOWED_PULLBACK_BASIS,
        )
        if ok:
            cleaned.pullback_basis = canonical
        else:
            original = cleaned.pullback_basis
            cleaned.pullback_basis = None
            notes.append(
                f"{PULLBACK_BASIS_UNRECOGNISED}: pullback_basis '{original}' "
                f"dropped — not in the controlled vocabulary "
                f"({sorted(ALLOWED_PULLBACK_BASIS)})."
            )

    # Rule 6 (Slice 3): pullback direction. Long ratings expect the
    # pullback to sit BELOW the close (a retracement before continuation);
    # short ratings expect ABOVE; Hold allows either side because a
    # Hold-with-bullish-lean-then-pullback pattern is legitimate. When
    # ``latest_close`` is missing we skip the directional check (mirrors
    # the trader-validator policy on missing close).
    if (
        cleaned.pullback_zone is not None
        and context.latest_close is not None
    ):
        close = context.latest_close
        pullback = cleaned.pullback_zone
        rating = cleaned.rating
        pullback_violation: Optional[str] = None
        if rating in _BULLISH_RATINGS and pullback >= close:
            pullback_violation = (
                f"pullback {pullback} is not below close {close} for a "
                f"{rating.value} rating — pullbacks on long ratings sit "
                f"below the current price."
            )
        elif rating in _BEARISH_RATINGS and pullback <= close:
            pullback_violation = (
                f"pullback {pullback} is not above close {close} for a "
                f"{rating.value} rating — pullbacks on short ratings sit "
                f"above the current price."
            )
        if pullback_violation is not None:
            cleaned.pullback_zone = None
            cleaned.pullback_basis = None
            notes.append(f"{PULLBACK_DIRECTION_VIOLATION}: {pullback_violation}")

    # Rule 7: orphan basis cleanup. Runs last so it sweeps any basis
    # orphaned by the earlier drops (currency mismatch, placeholder,
    # direction, vocabulary, pullback-direction).
    if cleaned.price_target_horizon is None and cleaned.target_basis:
        cleaned.target_basis = None
        notes.append(
            f"{TARGET_ORPHAN_BASIS}: target_basis dropped — orphan basis "
            f"without a paired price_target_horizon value."
        )
    if cleaned.pullback_zone is None and cleaned.pullback_basis:
        cleaned.pullback_basis = None
        notes.append(
            f"{TARGET_ORPHAN_BASIS}: pullback_basis dropped — orphan basis "
            f"without a paired pullback_zone value."
        )

    return ValidatedPortfolioDecision(decision=cleaned, notes=notes)


def _normalise_basis(
    value: Optional[str], allowed: frozenset[str],
) -> tuple[Optional[str], bool]:
    """Coerce a basis string to its canonical lowercase form if recognised.

    Returns ``(canonical_or_None, was_in_vocab)``. The case-insensitive
    match lets the model emit "DCF" and have it accepted as "dcf";
    unrecognised values surface as a named note in the caller rather than
    sneaking into the rendered report.
    """
    if value is None:
        return None, True
    canonical = value.strip().lower()
    if canonical in allowed:
        return canonical, True
    return None, False


def disagreement_usage_rate(
    decisions: list[ValidatedPortfolioDecision],
) -> float:
    """Fraction of validated decisions where the contract is suspended.

    "Suspended" means ``rating_target_disagreement`` is set to a value
    other than ``RatingTargetDisagreement.NONE``. Returns 0.0 for an empty
    list. Aggregators run this across a backtest or eval batch and treat
    rates above ~5% as a smell: too many "contrarian" decisions usually
    means the PM is dodging the directional rule rather than honestly
    naming a legitimate divergence.
    """
    if not decisions:
        return 0.0
    flagged = sum(
        1
        for v in decisions
        if v.decision.rating_target_disagreement is not None
        and v.decision.rating_target_disagreement
        != RatingTargetDisagreement.NONE
    )
    return flagged / len(decisions)


def render_pm_validation_notes(notes: list[str]) -> str:
    """Render PM-validator notes as a small markdown footer, or ``""``.

    Same shape as ``render_validation_notes`` for the Trader; kept as a
    named alias so callers in ``portfolio_manager.py`` read clearly and
    so the two footers can diverge later (e.g. PM-specific severity
    headers) without churn at the call sites.
    """
    return render_validation_notes(notes)


# ---------------------------------------------------------------------------
# Soft triangulation validator (Slice 4).
#
# The hard validator above DROPS fields when invariants fail. The soft
# triangulator never drops anything — it only emits advisory notes when
# the scorecard, rating, and target direction disagree. This surfaces
# weak signals (a strongly bullish scorecard paired with a Hold rating)
# that the directional contract on its own would not catch, without
# imposing a deterministic drop on what is ultimately a soft judgement
# call.
# ---------------------------------------------------------------------------

# Threshold for "strong" scorecard direction: 8 categories each scored
# -2..+2, so net spans -16..+16. |net| >= 4 is the cutoff at which the
# weight of evidence is firmly on one side; the PARACABLES case at -2
# is below this threshold (correctly, since the scorecard there really
# is roughly balanced — the Hold rating is reasonable on the scorecard
# alone; the hard contract still drops the target separately).
_SCORECARD_STRONG_THRESHOLD = 4


@dataclass
class TriangulatedPortfolioDecision:
    """Result of running the soft triangulator over a PortfolioDecision.

    ``decision`` is the input unchanged — the triangulator NEVER drops
    fields. ``notes`` carries any advisory messages flagged. Keep this
    distinct from ``ValidatedPortfolioDecision`` so callers know which
    notes are deterministic drops and which are advisory.
    """

    decision: PortfolioDecision
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.notes)


def _scorecard_net(decision: PortfolioDecision) -> Optional[int]:
    """Sum the scorecard's directional category scores.

    Returns ``None`` when no scorecard is attached. Confidence and the
    prose fields are not part of the sum — the net is a pure directional
    summary of the eight category scores.
    """
    sc = decision.scorecard
    if sc is None:
        return None
    return (
        sc.bull_case
        + sc.bear_case
        + sc.trend_technical
        + sc.fundamental_quality
        + sc.liquidity_risk
        + sc.catalyst_clarity
        + sc.macro_regime
        + sc.valuation
    )


def triangulate_portfolio_decision(
    decision: PortfolioDecision,
    context: PortfolioValidationContext,
) -> TriangulatedPortfolioDecision:
    """Run the SOFT triangulator over a PortfolioDecision.

    Emits two named advisory notes (no fields ever dropped):

    - ``SCORECARD_RATING_DIVERGENCE`` — the scorecard net is firmly on
      one side (``|net| >= _SCORECARD_STRONG_THRESHOLD``) but the rating
      does not match. Example: net +5 with a Hold rating.
    - ``SCORECARD_TARGET_DIVERGENCE`` — the scorecard net is firmly on
      one side but the horizon target points the other way relative to
      the latest close.

    Both checks require a scorecard; the target check also requires a
    known ``latest_close``. With either input missing the corresponding
    check is silently skipped.
    """
    notes: list[str] = []

    net = _scorecard_net(decision)
    if net is None:
        return TriangulatedPortfolioDecision(decision=decision, notes=notes)

    rating = decision.rating

    # Rule A: rating direction vs scorecard direction.
    if net >= _SCORECARD_STRONG_THRESHOLD and rating in (
        PortfolioRating.HOLD,
        PortfolioRating.UNDERWEIGHT,
        PortfolioRating.SELL,
    ):
        notes.append(
            f"{SCORECARD_RATING_DIVERGENCE}: scorecard net "
            f"{'+' if net > 0 else ''}{net} is strongly bullish but "
            f"rating is {rating.value} — consider upgrading the rating "
            f"or naming an offsetting risk factor that explains the gap."
        )
    elif net <= -_SCORECARD_STRONG_THRESHOLD and rating in (
        PortfolioRating.BUY,
        PortfolioRating.OVERWEIGHT,
        PortfolioRating.HOLD,
    ):
        notes.append(
            f"{SCORECARD_RATING_DIVERGENCE}: scorecard net "
            f"{net} is strongly bearish but rating is {rating.value} — "
            f"consider downgrading the rating or naming an offsetting "
            f"support factor that explains the gap."
        )

    # Rule B: target direction vs scorecard direction.
    close = context.latest_close
    target = decision.price_target_horizon
    if (
        close is not None
        and close > 0
        and target is not None
        and abs(net) >= _SCORECARD_STRONG_THRESHOLD
    ):
        if target > close * (1.0 + _DIRECTIONAL_EPSILON):
            target_dir = 1
        elif target < close * (1.0 - _DIRECTIONAL_EPSILON):
            target_dir = -1
        else:
            target_dir = 0
        scorecard_dir = 1 if net > 0 else -1
        if target_dir != 0 and target_dir != scorecard_dir:
            notes.append(
                f"{SCORECARD_TARGET_DIVERGENCE}: scorecard net "
                f"{'+' if net > 0 else ''}{net} points "
                f"{'bullish' if scorecard_dir > 0 else 'bearish'} but "
                f"price_target_horizon {target} relative to close "
                f"{close} points the other way — verify the target "
                f"reflects the scorecard's weight of evidence."
            )

    return TriangulatedPortfolioDecision(decision=decision, notes=notes)


def render_triangulation_notes(notes: list[str]) -> str:
    """Render triangulation notes as an advisory markdown footer.

    Kept distinct from ``render_pm_validation_notes`` so the reader sees
    which notes are deterministic drops (Validation Notes) and which are
    advisory soft signals (Triangulation Notes — no fields modified).
    """
    if not notes:
        return ""
    lines = ["**Triangulation Notes** (advisory; no fields modified):"]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)
