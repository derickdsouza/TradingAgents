"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.

The rendered decision is run through ``validate_portfolio_decision`` first
so semantically-invalid headlines (Buy + target below close, Hold + target
13% below close — the PARACABLES regression) are dropped before the
markdown is published, with a small ``Validation Notes`` footer naming
what was normalised.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_horizon_instruction,
    get_language_instruction,
)
from tradingagents.agents.utils.decision_contracts import (
    PortfolioValidationContext,
    _currency_from_ticker,
    _ticker_class,
    render_pm_validation_notes,
    render_triangulation_notes,
    triangulate_portfolio_decision,
    validate_portfolio_decision,
)
from tradingagents.agents.utils.evidence_ledger import render_evidence_ledger
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def _parse_latest_close(key_levels: str) -> Optional[float]:
    """Extract ``Latest close: <n>`` from the Key Price Levels block.

    Fallback for when the evidence ledger is absent. The same regex shape
    is used by the Trader; kept duplicated rather than imported to avoid
    a manager ↔ trader cycle.
    """
    if not key_levels:
        return None
    match = re.search(r"Latest close:\s*([0-9][0-9.,]*)", key_levels)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_trade_date(value: str) -> Optional[datetime]:
    """Parse a YYYY-MM-DD trade-date string. Returns None on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        key_levels = state.get("key_levels", "")
        market_regime = state.get("market_regime", "")
        ledger = state.get("evidence_ledger")
        ledger_block_text = render_evidence_ledger(ledger) if ledger else ""

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )
        if ledger_block_text:
            levels_block = f"\n\n{ledger_block_text}"
            regime_block = ""
        else:
            levels_block = f"\n\n{key_levels}" if key_levels else ""
            regime_block = f"\n\n{market_regime}" if market_regime else ""

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}{levels_block}{regime_block}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Price Target Semantics:**
- `price_target_horizon` answers: "If my rating plays out over the stated `time_horizon`, where is the price?" It MUST point the same direction as the rating: above current close for Buy/Overweight (≥3% above), below for Sell/Underweight (≥3% below), within a volatility-scaled band for Hold (typically ±5-20% depending on the underlying's volatility).
- `pullback_zone` answers: "Before that horizon target is reached, do I expect a meaningful retracement first?" If yes, name the level and its basis. If no, leave NULL. NEVER use this field as a substitute for `price_target_horizon`.
- `rating_target_disagreement` is the explicit escape hatch for legitimate contrarian positions: a Hold with a -15% target for a dividend-floor stock is fine if you name the reason. "I'm worried" is NOT a valid reason — downgrade the rating or use `pullback_zone` instead.
- Controlled vocabularies (use exactly these tokens):
  - `target_basis`: base_case, bear_case_skew, mean_reversion, range_bound, catalyst_neutral, dcf, peer_multiple, peg_at_consensus, technical_measured_move
  - `pullback_basis`: retest_breakout, fibonacci, prior_consolidation, moving_average, vwap_anchor, support_zone
  - `rating_target_disagreement`: dividend_floor, quality_premium, momentum_override, structural_optionality, merger_arb_floor, catalyst_neutral, none

Anti-patterns the validator will reject:
- Hold with target below close as a placeholder for "I'm worried" → either downgrade the rating, use `pullback_zone`, or set `rating_target_disagreement`.
- `price_target_horizon` equal to current close → asserts zero expected drift over the horizon. Use a basis of `catalyst_neutral` if intentional.
- Setting `price_target_horizon` to the Trader's stop-out level → stops belong to the Trader; the horizon target is your independent view of where price ENDS.

**Evidence Scorecard** — populate the structured `scorecard` field by scoring each named category on a -2 (strongly bearish) to +2 (strongly bullish) scale, with 0 meaning insufficient or balanced evidence. Categories: Bull Case, Bear Case, Trend / Technical, Fundamental Quality, Liquidity / Risk, Catalyst Clarity, Macro / Regime, Valuation. Set Confidence (Low / Medium / High) and write a one-to-two-sentence `rating_rationale` showing how the scorecard maps to the final rating. For Buy / Sell, name the specific evidence in `invalidating_evidence` that would downgrade or upgrade the call. For Hold, the `tie_breaker` field is REQUIRED — explain why the evidence is genuinely balanced rather than indecisive, and what would break the balance.

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_horizon_instruction()}{get_language_instruction()}"""

        # Build the validator context from the evidence ledger (preferred)
        # or the legacy Key Price Levels block (fallback). When neither
        # carries a usable close, the validator's loud-fail rule drops
        # any unverifiable target — better than publishing one anchored
        # to nothing.
        ledger_close: Optional[float] = None
        ledger_close_as_of: Optional[datetime] = None
        ledger_vol: Optional[float] = None
        if (
            ledger
            and ledger.latest_close
            and isinstance(ledger.latest_close.value, (int, float))
        ):
            ledger_close = float(ledger.latest_close.value)
            ledger_close_as_of = _parse_trade_date(
                ledger.latest_close.as_of or ""
            )
        # Slice 2: pull annualised volatility from the ledger's additive
        # ``extras`` shelf when an analyst has merged it in. No analyst
        # currently emits this key, but the seam is in place: when one
        # does, the Hold band immediately stops emitting
        # HOLD_BAND_UNCALIBRATED.
        if ledger and "annualised_volatility" in ledger.extras:
            fact = ledger.extras["annualised_volatility"]
            if isinstance(fact.value, (int, float)):
                ledger_vol = float(fact.value)

        latest_close = (
            ledger_close
            if ledger_close is not None
            else _parse_latest_close(key_levels)
        )
        trade_date_dt = _parse_trade_date(state.get("trade_date", "")) or datetime.utcnow()
        close_as_of = ledger_close_as_of or trade_date_dt

        # Slice 2: derive the close currency from the ticker suffix so
        # the validator's TARGET_CURRENCY_MISMATCH rule can fire on
        # cross-currency targets. Unsuffixed tickers default to USD.
        close_currency = _currency_from_ticker(state["company_of_interest"])

        # Slice 4: derive the coarse instrument class from the ticker so
        # the validator can reject point targets on indexes / FX / macro
        # rates / commodity futures (POINT_TARGET_INAPPROPRIATE).
        ticker_class = _ticker_class(state["company_of_interest"])

        validation_context = PortfolioValidationContext(
            trade_date=trade_date_dt,
            latest_close=latest_close,
            close_as_of=close_as_of,
            annualised_volatility=ledger_vol,
            close_currency=close_currency,
            ticker_class=ticker_class,
        )

        def _validated_render(decision: PortfolioDecision) -> str:
            # Slice 2: when the LLM doesn't fill target_currency, default
            # it to the close currency so the validator does NOT trip on
            # a None vs string mismatch. The currency field carries the
            # invariant; defaulting is the caller's job, not the
            # validator's.
            if (
                decision.price_target_horizon is not None
                and decision.target_currency is None
                and close_currency is not None
            ):
                decision = decision.model_copy(
                    update={"target_currency": close_currency}
                )
            validated = validate_portfolio_decision(decision, validation_context)
            md = render_pm_decision(validated.decision, latest_close=latest_close)
            footer = render_pm_validation_notes(validated.notes)
            # Slice 4: soft triangulation runs over the VALIDATED decision
            # (so notes apply to the post-drop state, not the raw LLM
            # output). The triangulator never modifies the decision; its
            # footer surfaces advisory divergences distinct from the
            # deterministic Validation Notes above.
            triangulated = triangulate_portfolio_decision(
                validated.decision, validation_context,
            )
            triangulation_footer = render_triangulation_notes(triangulated.notes)
            parts = [md]
            if footer:
                parts.append(footer)
            if triangulation_footer:
                parts.append(triangulation_footer)
            return "\n\n".join(parts)

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            _validated_render,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
