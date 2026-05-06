from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators.

Focus on: growth potential, competitive advantages, financial health, industry tailwinds, and a direct rebuttal of the bear's most recent point. Be conversational, not a bulleted list of facts.

**Anti-repetition rules — these are strict:**
- If a point appears in your prior turns (see `bull_history` below), do NOT restate it. Each turn must add a *new* data point or a specific counter to the bear's *latest* turn.
- Do NOT restate the analyst reports' findings — assume the reader has them. Reference them by name (e.g. "as the fundamentals report shows") and move on.
- Hard cap: 250 words per turn. Concise rebuttal beats long monologue.
- If you have no new material left, your turn is one sentence: "I rest on my prior arguments." Do not pad.

**Resources (do not paraphrase — cite when needed):**
- Market research report: {market_research_report}
- Social media sentiment report: {sentiment_report}
- Latest world affairs news: {news_report}
- Company fundamentals report: {fundamentals_report}

**Your prior turns (DO NOT REPEAT THESE):**
{bull_history if bull_history else "(none yet — this is round 1)"}

**Full debate so far:**
{history}

**Bear's most recent argument (rebut this specifically):**
{current_response}
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
