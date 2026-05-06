from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, weigh both sides and propose a balanced adjustment to the trader's plan that captures upside without ignoring risk.

**Trader's decision:** {trader_decision}

**Anti-repetition rules — these are strict:**
- If a point appears in your prior turns (see `neutral_history` below), do NOT restate it. Each turn must add a *new* data point or counter the aggressive/conservative's *latest* turn.
- Do NOT restate the analyst reports or trader plan — assume the room has read them. Cite by reference, do not summarize.
- Hard cap: 250 words per turn.
- If you have no new material, your turn is one sentence: "I rest on my prior arguments." Do not pad.

**Resources (cite, don't paraphrase):**
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**Your prior turns (DO NOT REPEAT):**
{neutral_history if neutral_history else "(none yet — this is round 1)"}

**Aggressive's latest:** {current_aggressive_response if current_aggressive_response else "(none yet)"}
**Conservative's latest:** {current_conservative_response if current_conservative_response else "(none yet)"}

Output conversationally, no special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
