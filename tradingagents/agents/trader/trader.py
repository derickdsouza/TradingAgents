"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_horizon_instruction,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        key_levels = state.get("key_levels", "")
        market_regime = state.get("market_regime", "")

        levels_block = f"\n\n{key_levels}" if key_levels else ""
        regime_block = f"\n\n{market_regime}" if market_regime else ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent. Convert the research plan into a concrete transaction. "
                    "Output ONLY the structured fields (action, entry, stop, sizing) plus a 1-2 sentence "
                    "rationale that REFERENCES the research plan — do NOT restate or summarize it. "
                    "Assume the reader has the analyst reports and research plan in front of them. "
                    "Anchor your entry/stop/target in the Key Price Levels block when one is supplied "
                    "— those are the deterministic levels and override any conflicting numbers in the "
                    "research plan. "
                    "ALWAYS populate `entry_price` with the deterministic price level at which a long "
                    "position is or would be ANCHORED — typically the most recent consolidation low, "
                    "a key moving-average cluster (50/200-DMA), or a pivot/AVWAP level visible in the "
                    "Key Price Levels block. NEVER use the latest close as a placeholder: latest close "
                    "is the *current* price, not the *entry* level the recommendation is anchored to. "
                    "For Buy, this is the planned entry. For Hold, it is the prior accumulation level "
                    "or pullback zone the position is anchored to / would re-enter at. For Sell, it is "
                    "the prior anchor level used as the cost reference for the exit. The report header "
                    "surfaces this number as the trade's anchor, so picking the latest close makes the "
                    "report look like the recommendation is to buy now at market — which is wrong on "
                    "Hold/Sell. Never leave `entry_price` blank."
                    + get_horizon_instruction()
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {company_name}. {instrument_context}{levels_block}{regime_block}\n\n"
                    f"Research plan (already in the reader's hands — reference, don't restate):\n"
                    f"{investment_plan}"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
