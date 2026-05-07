"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_horizon_instruction,
    get_language_instruction,
)
from tradingagents.agents.utils.evidence_ledger import render_evidence_ledger
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        key_levels = state.get("key_levels", "")
        market_regime = state.get("market_regime", "")
        ledger = state.get("evidence_ledger")
        ledger_block_text = render_evidence_ledger(ledger) if ledger else ""

        investment_debate_state = state["investment_debate_state"]

        # Prefer the structured ledger when populated; the legacy
        # key_levels / market_regime strings remain the fallback so
        # graphs without the analyst node still work.
        if ledger_block_text:
            levels_block = f"\n\n{ledger_block_text}"
            regime_block = ""
        else:
            levels_block = f"\n\n{key_levels}" if key_levels else ""
            regime_block = f"\n\n{market_regime}" if market_regime else ""

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}{levels_block}{regime_block}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Debate History:**
{history}

---
{get_horizon_instruction()}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
