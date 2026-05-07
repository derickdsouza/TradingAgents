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
from typing import Optional

from pydantic import BaseModel, Field


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
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
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


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.scorecard is not None:
        parts.extend(["", render_evidence_scorecard(decision.scorecard)])
    return "\n".join(parts)
