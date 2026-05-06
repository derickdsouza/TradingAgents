from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Present a well-reasoned argument emphasizing risks, challenges, and negative indicators.

Focus on: financial instability, macro threats, competitive weaknesses, negative indicators, and a direct rebuttal of the bull's most recent point. Be conversational, not a bulleted list of facts.

**Anti-repetition rules — these are strict:**
- If a point appears in your prior turns (see `bear_history` below), do NOT restate it. Each turn must add a *new* data point or a specific counter to the bull's *latest* turn.
- Do NOT restate the analyst reports' findings — assume the reader has them. Reference them by name (e.g. "as the fundamentals report shows") and move on.
- Hard cap: 250 words per turn. Concise rebuttal beats long monologue.
- If you have no new material left, your turn is one sentence: "I rest on my prior arguments." Do not pad.

**Numerical-fidelity rules — also strict:**
- When you cite a price, moving average, ratio, percentage, OI, RVOL, ADX,
  growth rate, or any other numeric value, copy it VERBATIM from the named
  source report. Do not round, restate in different units, or invent a level.
- Tag the source on first use, e.g. "per market_report: 200-DMA = 358.53"
  or "per fundamentals_report: FY25 OCF = INR -128.8 Cr". The trader, the
  research manager, the risk debaters, and the portfolio manager all read
  this transcript downstream — they cannot cross-check against the source
  reports, so any number you alter or hallucinate becomes load-bearing
  fiction the rest of the pipeline will commit to.
- If a number you need is not in the analyst reports, say so explicitly
  ("the reports do not quote a current 20-day high; I am working off the
  RVOL/breakout flags only") rather than fabricating one.
- Quote currency must match the source: INR Cr / lakh for Indian-listing
  financials, USD/millions only for genuinely international references
  (Brent, US gasoline, FX, foreign peers). Do not unit-convert mid-debate.

**Resources (do not paraphrase — cite when needed):**
- Market research report: {market_research_report}
- Social media sentiment report: {sentiment_report}
- Latest world affairs news: {news_report}
- Company fundamentals report: {fundamentals_report}

**Your prior turns (DO NOT REPEAT THESE):**
{bear_history if bear_history else "(none yet — this is round 1)"}

**Full debate so far:**
{history}

**Bull's most recent argument (rebut this specifically):**
{current_response}
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
