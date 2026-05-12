"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import re
from typing import Optional

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_horizon_instruction,
    get_language_instruction,
)
from tradingagents.agents.utils.decision_contracts import (
    TraderValidationContext,
    render_validation_notes,
    validate_trader_proposal,
)
from tradingagents.agents.utils.evidence_ledger import render_evidence_ledger
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def _parse_latest_close(key_levels: str) -> Optional[float]:
    """Extract the deterministic ``Latest close: <n>`` value from key_levels.

    The Key Price Levels block is the agent's reference price source. The
    same regex shape is used by the report renderer in ``cli/main.py``;
    keeping it duplicated rather than importing avoids a CLI ↔ agents
    cycle. Returns ``None`` when the block is absent or unparseable —
    the validator treats that as "no directional check possible" rather
    than failing.
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


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        key_levels = state.get("key_levels", "")
        market_regime = state.get("market_regime", "")
        ledger = state.get("evidence_ledger")
        ledger_block_text = render_evidence_ledger(ledger) if ledger else ""

        if ledger_block_text:
            levels_block = f"\n\n{ledger_block_text}"
            regime_block = ""
        else:
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
                    "ALWAYS populate `entry_price` AND `entry_basis`: entry_price is the "
                    "deterministic price level the position is or would be ANCHORED at — typically the "
                    "most recent consolidation low, a key moving-average cluster (50/200-DMA), or a "
                    "pivot/AVWAP level visible in the Key Price Levels block; entry_basis is a one-line "
                    "label naming what the level is (e.g. '50-DMA', '50-DMA/200-DMA cluster midpoint "
                    "(psychological)', 'AVWAP from 52-week low'). NEVER use the latest close as a "
                    "placeholder for entry_price: latest close is the *current* price, not the *entry* "
                    "level. For Buy, entry_price is the planned entry. For Hold, it is the prior "
                    "accumulation level or pullback zone the position is anchored to / would re-enter "
                    "at. For Sell, it is the prior anchor level used as the cost reference for the "
                    "exit. "
                    "STOPS — split into two distinct concepts: (a) `stop_initial` + "
                    "`stop_initial_basis` is the FIXED line in the sand from entry where the trade "
                    "thesis is invalidated. Either a technical stop (just below 20-day low / swing low "
                    "/ 200-DMA — name it in stop_initial_basis as e.g. 'just below 20-day low') OR a "
                    "thesis-break stop (round number representing fundamental deterioration — name it "
                    "as e.g. 'thesis-break (fundamental, fixed)'). **Place the stop with a BUFFER below "
                    "the structural level you anchor to — never AT the level itself. A stop at the "
                    "200-DMA is triggered by routine retests; a stop ~3-5% below the 200-DMA only "
                    "fires on a genuine break. Buffer rule of thumb: swing 3%, position 4%, long-term "
                    "5%.** (b) `stop_trailing` + "
                    "`stop_trailing_basis` is the OPTIONAL DYNAMIC trailing stop that moves up as the "
                    "trade moves in favour, anchored to a trailing reference like 'Chandelier Exit "
                    "(ATR, dynamic)', 'AVWAP-52wL', or 'rising 50-DMA'. Many real swing setups carry "
                    "BOTH simultaneously: a tight technical stop_initial AND a dynamic stop_trailing. "
                    "If the Key Price Levels or analyst reports surface a Chandelier Exit, AVWAP, or "
                    "similar dynamic level, populate stop_trailing with it. The basis labels are "
                    "surfaced verbatim in the report header so the reader sees what each number "
                    "actually means."
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

        ledger_close: Optional[float] = None
        if ledger and ledger.latest_close and isinstance(ledger.latest_close.value, (int, float)):
            ledger_close = float(ledger.latest_close.value)
        # Slice 7 (74l): thread cited structural levels into the validator
        # so a stop placed AT a level (zero buffer) gets flagged. We only
        # surface the levels typically used as STOP anchors — SMAs and
        # recent lows for longs, recent highs for shorts. The validator
        # treats the rule symmetrically so labelling each as "level" is
        # enough; the basis label on the stop itself stays the reader's
        # signal of intent.
        support_levels: list[tuple[str, float]] = []
        if ledger:
            for label, fact in (
                ("200-DMA", ledger.sma_200),
                ("50-DMA", ledger.sma_50),
                ("52w-low", ledger.low_52w),
                ("20d-low", ledger.low_20d),
                ("52w-high", ledger.high_52w),
                ("20d-high", ledger.high_20d),
            ):
                if fact and isinstance(fact.value, (int, float)):
                    support_levels.append((label, float(fact.value)))
        from tradingagents.dataflows.config import get_config as _get_cfg
        horizon_key = (_get_cfg().get("trading_horizon") or "").lower() or None
        validation_context = TraderValidationContext(
            latest_close=ledger_close if ledger_close is not None else _parse_latest_close(key_levels),
            support_levels=tuple(support_levels),
            horizon=horizon_key,
        )

        def _validated_render(proposal: TraderProposal) -> str:
            validated = validate_trader_proposal(proposal, validation_context)
            md = render_trader_proposal(validated.proposal)
            footer = render_validation_notes(validated.notes)
            return f"{md}\n\n{footer}" if footer else md

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            _validated_render,
            "Trader",
            schema=TraderProposal,
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
