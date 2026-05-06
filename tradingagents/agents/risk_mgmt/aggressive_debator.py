from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Aggressive Risk Analyst, champion the high-reward path. Focus on upside, growth, and where the conservative/neutral views are leaving alpha on the table.

**Trader's decision:** {trader_decision}

**Anti-repetition rules — these are strict:**
- If a point appears in your prior turns (see `aggressive_history` below), do NOT restate it. Each turn must add a *new* data point or counter the conservative/neutral's *latest* turn.
- Do NOT restate the analyst reports or trader plan — assume the room has read them. Cite by reference, do not summarize.
- Hard cap: 250 words per turn.
- If you have no new material, your turn is one sentence: "I rest on my prior arguments." Do not pad.

**Resources (cite, don't paraphrase):**
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**Your prior turns (DO NOT REPEAT):**
{aggressive_history if aggressive_history else "(none yet — this is round 1)"}

**Conservative's latest:** {current_conservative_response if current_conservative_response else "(none yet)"}
**Neutral's latest:** {current_neutral_response if current_neutral_response else "(none yet)"}

Output conversationally, no special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
