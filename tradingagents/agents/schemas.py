"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class Confidence(str, Enum):
    """Manager confidence in the rating, used by the Evidence Scorecard."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RatingTargetDisagreement(str, Enum):
    """Named reasons the horizon target may legitimately contradict the rating.

    Slice 3 of the PM decision contract introduces an explicit escape hatch
    for contrarian positions that the directional rule would otherwise drop.
    When set to anything other than ``NONE`` the validator suspends the
    directional contract for that decision and surfaces the value as a
    rendered ``Disagreement Rationale`` line. The vocabulary is closed so an
    aggregator can count usage frequency across runs and flag overuse.
    """

    DIVIDEND_FLOOR = "dividend_floor"
    QUALITY_PREMIUM = "quality_premium"
    MOMENTUM_OVERRIDE = "momentum_override"
    STRUCTURAL_OPTIONALITY = "structural_optionality"
    MERGER_ARB_FLOOR = "merger_arb_floor"
    CATALYST_NEUTRAL = "catalyst_neutral"
    NONE = "none"


# ---------------------------------------------------------------------------
# Evidence Scorecard
# ---------------------------------------------------------------------------


class EvidenceScorecard(BaseModel):
    """Compact, named scoring of the evidence categories that drive the rating.

    Both the Research Manager and the Portfolio Manager fill this when
    issuing a rating so the trade-off between competing categories is
    explicit instead of buried in prose. Each category scores from -2
    (strongly bearish) to +2 (strongly bullish); 0 means insufficient
    or balanced evidence. The rendered scorecard appears in the saved
    report and lets evaluation harnesses pin rating drivers.
    """

    bull_case: int = Field(
        ge=-2, le=2,
        description=(
            "Score for the bull side of the debate / argument. -2 to +2. "
            "Higher = stronger bull conviction surfaced by the evidence."
        ),
    )
    bear_case: int = Field(
        ge=-2, le=2,
        description=(
            "Score for the bear side of the debate / argument. -2 to +2. "
            "More negative = stronger bear conviction surfaced by the evidence."
        ),
    )
    trend_technical: int = Field(
        ge=-2, le=2,
        description=(
            "Trend / technical setup quality. -2 to +2. Reads MA structure, "
            "breakout/breakdown signals, RS, and pattern integrity."
        ),
    )
    fundamental_quality: int = Field(
        ge=-2, le=2,
        description=(
            "Fundamental quality (margins, growth, returns on capital, "
            "balance-sheet strength). -2 to +2."
        ),
    )
    liquidity_risk: int = Field(
        ge=-2, le=2,
        description=(
            "Liquidity and balance-sheet risk. -2 (severe risk) to +2 "
            "(fortress balance sheet)."
        ),
    )
    catalyst_clarity: int = Field(
        ge=-2, le=2,
        description=(
            "Clarity and proximity of catalysts. -2 to +2. Higher = clearer "
            "near-term drivers (earnings, product launches, macro events)."
        ),
    )
    macro_regime: int = Field(
        ge=-2, le=2,
        description=(
            "Macro / market-regime support. -2 to +2. Reads broad-market "
            "trend, sector regime, and FII/DII flow context."
        ),
    )
    valuation: int = Field(
        ge=-2, le=2,
        description=(
            "Valuation tilt vs the thesis. -2 (stretched) to +2 (deeply "
            "supportive). Use 0 when valuation is fair or unclear."
        ),
    )
    confidence: Confidence = Field(
        description="Overall confidence in the rating: Low / Medium / High.",
    )
    rating_rationale: str = Field(
        description=(
            "Concise explanation of how the scorecard maps to the assigned "
            "rating tier. One to two sentences."
        ),
    )
    tie_breaker: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when the rating is Hold: explicit explanation of why "
            "the evidence is genuinely balanced rather than indecisive. "
            "Optional otherwise."
        ),
    )
    invalidating_evidence: str = Field(
        description=(
            "What would change the rating: for Buy/Sell name the specific "
            "evidence that would downgrade/upgrade; for Hold name the trigger "
            "that would break the balance."
        ),
    )


def render_evidence_scorecard(sc: EvidenceScorecard) -> str:
    """Render an EvidenceScorecard as a stable markdown block.

    The block keeps category names and signed scores in a fixed order so
    reports compare cleanly across runs and tests can pin specific lines.
    """

    def _signed(n: int) -> str:
        return f"+{n}" if n > 0 else str(n)

    lines = [
        "**Scorecard**:",
        f"- Bull Case: {_signed(sc.bull_case)}",
        f"- Bear Case: {_signed(sc.bear_case)}",
        f"- Trend / Technical: {_signed(sc.trend_technical)}",
        f"- Fundamental Quality: {_signed(sc.fundamental_quality)}",
        f"- Liquidity / Risk: {_signed(sc.liquidity_risk)}",
        f"- Catalyst Clarity: {_signed(sc.catalyst_clarity)}",
        f"- Macro / Regime: {_signed(sc.macro_regime)}",
        f"- Valuation: {_signed(sc.valuation)}",
        f"- Confidence: {sc.confidence.value}",
        "",
        f"**Rating Rationale**: {sc.rating_rationale}",
    ]
    if sc.tie_breaker:
        lines.extend(["", f"**Tie Breaker**: {sc.tie_breaker}"])
    lines.extend(["", f"**Invalidating Evidence**: {sc.invalidating_evidence}"])
    return "\n".join(lines)


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )
    scorecard: Optional[EvidenceScorecard] = Field(
        default=None,
        description=(
            "Optional structured evidence scorecard that captures how named "
            "categories were weighed before assigning the rating. Fill this "
            "whenever evidence categories can be rated; the rendered block "
            "is shown in the report and consumed by evaluation harnesses."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    parts = [
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ]
    if plan.scorecard is not None:
        parts.extend(["", render_evidence_scorecard(plan.scorecard)])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.

    Stops are split into two distinct concepts that real traders manage
    separately: ``stop_initial`` is the fixed line in the sand from entry
    (technical or thesis-break), ``stop_trailing`` is the dynamic exit
    once the trade is in profit (typically Chandelier Exit / AVWAP /
    rising MA). Each price field has a paired ``_basis`` field naming
    exactly what level it is anchored to, so the report can show
    ``Initial Stop: 313 (just below 20-day low)`` rather than a bare
    number that reads as either a technical or psychological stop.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description=(
            "Entry-anchor price in the instrument's quote currency. The "
            "deterministic level the recommendation is anchored to — a recent "
            "consolidation low, a moving-average cluster, a pivot, an AVWAP, "
            "or a round psychological level. NEVER use the latest close as a "
            "placeholder: latest close is the *current* price, not the *entry* "
            "level."
        ),
    )
    entry_basis: Optional[str] = Field(
        default=None,
        description=(
            "One-line label naming what `entry_price` is anchored to. Examples: "
            "'50-DMA', '50-DMA/200-DMA cluster midpoint (psychological)', "
            "'prior swing high', 'AVWAP from 52-week low', 'cup-and-handle "
            "pivot'. ALWAYS populate when `entry_price` is set — the report "
            "header surfaces it so the reader sees what the number means."
        ),
    )
    stop_initial: Optional[float] = Field(
        default=None,
        description=(
            "Initial stop-loss price — the FIXED level at which the trade "
            "thesis is invalidated and the position is closed. The line in "
            "the sand from entry. Either a technical level (just below 20-day "
            "low / swing low / 200-DMA) or a thesis-break level (round number "
            "representing fundamental deterioration). "
            "DIRECTIONAL CONSTRAINT: for a long-side action (Buy / Hold of an "
            "existing long), this number MUST be BELOW the current close. For "
            "a Sell / short, it MUST be ABOVE. A level above current price on "
            "a long is a target or upgrade trigger, NOT a stop — leave the "
            "field NULL rather than misuse it."
        ),
    )
    stop_initial_basis: Optional[str] = Field(
        default=None,
        description=(
            "One-line label naming what `stop_initial` is anchored to. "
            "Examples: 'just below 20-day low', 'below 200-DMA buffer', "
            "'thesis-break (fundamental, fixed)'. ALWAYS populate when "
            "`stop_initial` is set."
        ),
    )
    stop_trailing: Optional[float] = Field(
        default=None,
        description=(
            "Optional dynamic trailing-stop price — moves up as the trade "
            "moves in favour. Typically anchored to Chandelier Exit (ATR-"
            "based), AVWAP-52wL, or a rising 50-DMA. DISTINCT from "
            "`stop_initial`: trailing stops are DYNAMIC (recompute each bar "
            "and ratchet in the direction of profit); initial stops are "
            "FIXED. Many real swing setups carry both simultaneously. "
            "STATIC-VS-DYNAMIC RULE: if the level does not ratchet (a 'hard "
            "floor', a fixed thesis-break level, a round-number panic exit, "
            "or any single static price below entry that doesn't move), it "
            "belongs in `stop_initial`'s reasoning, NOT here. Only populate "
            "`stop_trailing` when the exit is genuinely path-dependent on "
            "future price action. "
            "DIRECTIONAL CONSTRAINT: for a long-side action (Buy / Hold of an "
            "existing long), this number MUST be BELOW the current close. For "
            "a Sell / short, it MUST be ABOVE. NEVER use this field for an "
            "upgrade trigger, breakout-confirmation level, prior swing high, "
            "or upside target — those are NOT stops. If no dynamic exit "
            "applies (e.g. Hold with no incremental dynamic exit), leave NULL."
        ),
    )
    stop_trailing_basis: Optional[str] = Field(
        default=None,
        description=(
            "One-line label naming what `stop_trailing` is anchored to. "
            "Examples: 'Chandelier Exit (ATR, dynamic)', 'AVWAP-52wL', "
            "'rising 50-DMA'. ALWAYS populate when `stop_trailing` is set."
        ),
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    Each price field is rendered with its paired basis label inline as
    ``**Label**: <value> — _<basis>_`` when the basis is present, so the
    rendered prose carries the same information the structured schema
    captured.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """

    def _line(label: str, value, basis: Optional[str]) -> str:
        line = f"**{label}**: {value}"
        if basis:
            line += f" — _{basis}_"
        return line

    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", _line("Entry Price", proposal.entry_price, proposal.entry_basis)])
    if proposal.stop_initial is not None:
        parts.extend(["", _line("Initial Stop", proposal.stop_initial, proposal.stop_initial_basis)])
    if proposal.stop_trailing is not None:
        parts.extend(["", _line("Trailing Stop", proposal.stop_trailing, proposal.stop_trailing_basis)])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target_horizon: Optional[float] = Field(
        default=None,
        description=(
            "Optional price target the position is anchored to over the "
            "stated `time_horizon` (or the prompt's default horizon when "
            "`time_horizon` is empty). In the instrument's quote currency. "
            "DIRECTIONAL CONSTRAINT: for bullish ratings (Buy / Overweight) "
            "this number MUST be MEANINGFULLY ABOVE the current close — a "
            "target below close is a Sell signal, not a Buy target. For "
            "bearish ratings (Underweight / Sell) it MUST be MEANINGFULLY "
            "BELOW the current close. For Hold the target sits within a "
            "narrow band around the close (the position is anchored, not "
            "directional). 'Meaningfully' means more than statistical noise "
            "(roughly ±3% of close). "
            "NEVER use the latest close itself as a placeholder: latest "
            "close is the *current* price, not a *target*. "
            "NEVER use this field for a pullback / re-entry level: that is "
            "structurally different from a horizon target — a target is a "
            "level the position is *pointed at*, not a level it would buy "
            "back at. Leave NULL when no defensible target exists."
        ),
    )
    target_basis: Optional[str] = Field(
        default=None,
        description=(
            "One-line label naming what `price_target_horizon` is anchored "
            "to (e.g. 'DCF', 'PE multiple', 'analyst consensus', 'range "
            "midpoint'). ALWAYS populate when `price_target_horizon` is "
            "set so the reader sees what the number means."
        ),
    )
    target_range_low: Optional[float] = Field(
        default=None,
        description=(
            "Lower bound of the horizon target range. Pair with `target_range_high` "
            "and `price_target_horizon` (median). Point targets at 12-month horizons "
            "are false precision; prefer ranges unless you have a specific basis "
            "(DCF base case, peer multiple) anchored to a single number. Index, FX, "
            "and macro tickers REQUIRE range form — point targets on those classes "
            "are rejected by the validator."
        ),
    )
    target_range_high: Optional[float] = Field(
        default=None,
        description=(
            "Upper bound of the horizon target range. Pair with `target_range_low` "
            "and `price_target_horizon` (median). Required when `target_range_low` "
            "is set."
        ),
    )
    target_currency: Optional[str] = Field(
        default=None,
        description=(
            "Currency of the horizon target. Optional ISO-4217 (USD, INR, "
            "GBP, HKD, CAD, AUD, JPY, EUR, ...). Default-derived from the "
            "ticker suffix when None. The validator rejects the target when "
            "this currency does not match the currency of the latest close."
        ),
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    scorecard: Optional[EvidenceScorecard] = Field(
        default=None,
        description=(
            "Optional structured evidence scorecard mirroring the Research "
            "Manager's; fill this so the final rating is grounded in named "
            "evidence categories instead of only narrative synthesis."
        ),
    )
    rating_target_disagreement: Optional[RatingTargetDisagreement] = Field(
        default=None,
        description=(
            "Names a legitimate reason the horizon target may contradict the "
            "rating direction. When set to anything other than 'none', the "
            "directional contract is SUSPENDED for this decision and a "
            "'Disagreement Rationale' line is rendered in the report. Use this "
            "honestly: 'I'm worried' is NOT a valid disagreement — downgrade "
            "the rating or use `pullback_zone` instead. Audit fires when this "
            "is set on >5% of decisions. Allowed values: dividend_floor "
            "(Hold/Buy with target below close because of dividend yield "
            "support), quality_premium (Buy/Overweight despite target slightly "
            "below close because of compounding quality), momentum_override "
            "(Buy/Overweight on momentum despite stretched valuation), "
            "structural_optionality (Hold with bearish target because of "
            "takeover/breakup optionality), merger_arb_floor (Buy/Hold with "
            "narrow upside because of deal-close arbitrage), catalyst_neutral "
            "(target ≈ close because the position is a binary event play)."
        ),
    )
    pullback_zone: Optional[float] = Field(
        default=None,
        description=(
            "Optional NEAR-TERM retracement level the PM expects price to "
            "visit BEFORE the horizon thesis plays out. Distinct from "
            "`price_target_horizon` which is the END-OF-HORIZON expected "
            "price. Must be below current close for long ratings "
            "(Buy/Overweight/Hold with bullish lean); may be above close for "
            "short ratings or for Hold-with-upside-then-pullback patterns. "
            "Pair with `pullback_basis`."
        ),
    )
    pullback_basis: Optional[str] = Field(
        default=None,
        description=(
            "One-line label naming what `pullback_zone` is anchored to. "
            "Controlled vocabulary: retest_breakout, fibonacci, "
            "prior_consolidation, moving_average, vwap_anchor, support_zone. "
            "ALWAYS populate when `pullback_zone` is set."
        ),
    )
    target_committed_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD) when ``price_target_horizon`` was first "
            "published. Snapshotted by the renderer from ``state['trade_date']`` "
            "— NEVER set by the LLM. Used by Slice 5's target-drift validator "
            "on subsequent runs to detect a stale carry-forward target."
        ),
    )
    committed_close: Optional[float] = Field(
        default=None,
        description=(
            "Latest close at the moment ``target_committed_at`` was "
            "snapshotted. Snapshotted by the renderer — NEVER set by the "
            "LLM. Used by ``validate_target_drift`` on subsequent runs to "
            "compute how far the current close has drifted from the price "
            "at which the target was originally committed."
        ),
    )
    target_drift_threshold_pct: Optional[float] = Field(
        default=0.15,
        description=(
            "Fractional drift threshold (default 0.15 = 15%). When "
            "``|current_close - committed_close| / committed_close`` exceeds "
            "this value on the next run, the PM is required to reaffirm or "
            "revise the target rather than carry it forward implicitly."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_price_target(cls, data: Any) -> Any:
        """Coerce legacy ``price_target`` payloads into ``price_target_horizon``.

        Memory-log entries written before Slice 1 of the PM decision
        contract carried ``price_target`` as the key. New code reads
        ``price_target_horizon``. When both are supplied the new name wins
        — callers that have already adopted the new schema are not silently
        downgraded by stale data still carrying the old key. Removable
        once memory-log entries from before the migration roll out of the
        retention window.
        """
        if not isinstance(data, dict):
            return data
        if "price_target" in data and "price_target_horizon" not in data:
            data["price_target_horizon"] = data.pop("price_target")
        elif "price_target" in data:
            # Both present: the new name takes precedence.
            data.pop("price_target")
        return data


def render_pm_decision(
    decision: PortfolioDecision,
    latest_close: Optional[float] = None,
) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.

    Slice 2: when ``latest_close`` is supplied alongside a
    ``price_target_horizon``, an ``**Expected Return**`` line is emitted
    directly below the dual target lines as a signed percentage rounded
    to one decimal. This gives the reader and the evaluation harness the
    directional check in plain text without recomputing it from the
    target and close numbers.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target_horizon is not None:
        # MIGRATION SHIM (Slice 1 of PM decision contract): emit BOTH the
        # legacy ``Price Target`` and the new ``Horizon Target`` labels so
        # existing CLI parsers (cli/main.py `_build_trade_setup_block`,
        # `_ENSEMBLE_FIELDS["price_target"]`) keep matching while readers
        # learn the new label. REMOVE the legacy line once Slice 3 ships
        # OR 4 weeks pass (deadline ~2026-06-08), whichever first. Both
        # labels read from the single source of truth ``price_target_horizon``.
        target = decision.price_target_horizon
        target_line = f"**Price Target**: {target}"
        horizon_line = f"**Horizon Target**: {target}"
        if decision.target_basis:
            target_line += f" — _{decision.target_basis}_"
            horizon_line += f" — _{decision.target_basis}_"
        parts.extend(["", target_line, "", horizon_line])
        # Slice 5: target vintage. Surfacing the committed-at date and the
        # close at commit time lets the reader (and any aggregator) see how
        # fresh the target is and triggers the drift-reaffirmation directive
        # on the NEXT run when the current close has moved off the
        # committed close. Rendered only when BOTH fields are present —
        # neither field alone is meaningful.
        if (
            decision.target_committed_at is not None
            and decision.committed_close is not None
        ):
            parts.extend([
                "",
                f"**Target Vintage**: committed "
                f"{decision.target_committed_at} at "
                f"{decision.committed_close}",
            ])
    # Slice 4: Target Range line renders directly below the horizon target
    # when both bounds are set AND the range is non-trivial (i.e. bounds
    # differ from each other and from the horizon, when present). A trivial
    # range adds nothing the horizon target doesn't already say.
    if (
        decision.target_range_low is not None
        and decision.target_range_high is not None
    ):
        low = decision.target_range_low
        high = decision.target_range_high
        is_trivial = (
            low == high
            and decision.price_target_horizon is not None
            and low == decision.price_target_horizon
        )
        if not is_trivial:
            parts.extend(["", f"**Target Range**: {low} – {high}"])
    # Expected Return: prefer the horizon target as the reference when set,
    # otherwise fall back to the midpoint of the range. Either way the
    # reader gets the signed-pct directional check without recomputing.
    if latest_close is not None and latest_close > 0:
        reference: Optional[float] = None
        if decision.price_target_horizon is not None:
            reference = decision.price_target_horizon
        elif (
            decision.target_range_low is not None
            and decision.target_range_high is not None
        ):
            reference = (
                decision.target_range_low + decision.target_range_high
            ) / 2.0
        if reference is not None:
            pct = (reference - latest_close) / latest_close * 100.0
            sign = "+" if pct >= 0 else ""
            parts.extend(["", f"**Expected Return**: {sign}{pct:.1f}%"])
    # Slice 3: pullback zone renders independently of the horizon target
    # — a Hold-with-named-pullback may have no horizon target but still
    # surface a defensible retracement level.
    if decision.pullback_zone is not None:
        pullback_line = f"**Pullback Zone**: {decision.pullback_zone}"
        if decision.pullback_basis:
            pullback_line += f" — _{decision.pullback_basis}_"
        parts.extend(["", pullback_line])
    # Slice 3: surface the disagreement rationale so the reader (and an
    # aggregator counting usage) sees the named reason the directional
    # contract was suspended. ``NONE`` is the explicit "no disagreement"
    # value and is not rendered.
    if (
        decision.rating_target_disagreement is not None
        and decision.rating_target_disagreement != RatingTargetDisagreement.NONE
    ):
        parts.extend([
            "",
            f"**Disagreement Rationale**: "
            f"{decision.rating_target_disagreement.value}",
        ])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.scorecard is not None:
        parts.extend(["", render_evidence_scorecard(decision.scorecard)])
    return "\n".join(parts)
