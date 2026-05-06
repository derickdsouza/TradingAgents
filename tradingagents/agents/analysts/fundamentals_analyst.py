from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_analyst_horizon_instruction,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_horizon,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.agents.utils.fundamental_data_tools import get_shareholding_pattern
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.nse_client import is_indian_ticker


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
        ]
        if is_indian_ticker(ticker):
            tools.append(get_shareholding_pattern)

        lookback_phrase = get_horizon()["lookback_phrase"]
        system_message = (
            f"You are a researcher tasked with analyzing fundamental information over {lookback_phrase} about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " When you have all the data you need and are producing the final report, begin your response directly with the report content (e.g. a heading or the first analytical paragraph). Do NOT preface the report with sentences like 'Now I have all the data needed.' or 'Let me compile the analysis.' — those narrator-style intros are saved verbatim into the report file."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis; `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements; and `get_insider_transactions` to surface insider buying/selling that often precedes material moves."
            + (
                " For this Indian ticker, also call `get_shareholding_pattern` —"
                " quarterly promoter %, public %, and current pledge %."
                " Promoter pledge above 10% is a meaningful risk signal and"
                " a falling promoter % over multiple quarters often signals"
                " institutional confidence concerns."
                if is_indian_ticker(ticker) else ""
            )
            + get_analyst_horizon_instruction()
            + get_language_instruction(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Your role is fundamental analysis ONLY — do NOT output a final BUY/HOLD/SELL"
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
