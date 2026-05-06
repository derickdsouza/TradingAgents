from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_horizon_instruction,
    get_global_news,
    get_horizon,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.news_data_tools import (
    get_corporate_announcements,
    get_india_macro,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.nse_client import is_indian_ticker


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_news,
            get_global_news,
        ]
        if is_indian_ticker(ticker):
            tools.append(get_corporate_announcements)
            tools.append(get_india_macro)

        lookback_phrase = get_horizon()["lookback_phrase"]
        india_clause = (
            " For this Indian ticker, also call get_corporate_announcements(ticker, look_back_days)"
            " — this surfaces NSE-filed catalysts (board meetings, results, dividends,"
            " promoter pledge changes, insider trades under SEBI Reg 7(2), bulk/block deals)"
            " that are not in Yahoo news but routinely move Indian stocks. Also call"
            " get_india_macro() — INR/USD, Brent, Nifty levels, India VIX, and today's"
            " FII vs DII cash-market net flow. Indian equities move heavily on these"
            " numeric drivers and they're not visible in headline news."
            if is_indian_ticker(ticker) else ""
        )
        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over {lookback_phrase}. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(ticker, start_date, end_date) for company-specific news, and get_global_news(curr_date, look_back_days, limit, ticker) for broader macroeconomic news. Always pass the ticker under analysis to get_global_news — for Indian tickers (.NS/.BO) this switches the macro query set to RBI/CPI/FII-DII/INR-oil, which is what actually moves Indian markets."
            + india_clause
            + " Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + " When you have all the data you need and are producing the final report, begin your response directly with the report content (e.g. a heading or the first analytical paragraph). Do NOT preface the report with sentences like 'Now I have all the data needed.' or 'Let me compile the analysis.' — those narrator-style intros are saved verbatim into the report file."
            + get_analyst_horizon_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Your role is news/macro analysis ONLY — do NOT output a final BUY/HOLD/SELL"
                    " transaction proposal; the Portfolio Manager produces the verdict downstream."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
