from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Always emits a directive — even for English — because Chinese-trained
    providers (e.g. GLM) sometimes code-switch into CJK tokens mid-sentence
    when no explicit anti-code-switch instruction is present. Applied to
    every agent whose output reaches the saved report — analysts,
    researchers, debaters, research manager, trader, and portfolio manager
    — so a non-English run produces a fully localized report rather than a
    mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English").strip()
    if lang.lower() == "english":
        return (
            " Write your entire response in English only. "
            "Do not insert words or characters from any other script "
            "(no Chinese / CJK, no Devanagari, no Arabic, etc.)."
        )
    return (
        f" Write your entire response in {lang} only. "
        f"Do not code-switch into any other language mid-sentence."
    )


HORIZONS = {
    "swing": {
        "label": "swing trade",
        "holding_period": "2-6 weeks",
        "lookback_days": 7,
        "lookback_phrase": "the past week",
    },
    "position": {
        "label": "position trade",
        "holding_period": "3-6 months",
        "lookback_days": 30,
        "lookback_phrase": "the past 30 days",
    },
    "long-term": {
        "label": "long-term investment",
        "holding_period": "12+ months",
        "lookback_days": 90,
        "lookback_phrase": "the past 90 days",
    },
}


def get_horizon() -> dict:
    """Return the active horizon profile from config (default: position)."""
    from tradingagents.dataflows.config import get_config
    key = (get_config().get("trading_horizon") or "position").lower()
    return HORIZONS.get(key, HORIZONS["position"])


def get_horizon_instruction() -> str:
    """Sentence appended to decision-agent prompts (Trader, RM, PM).

    Anchors recommendations, price targets, and time_horizon fields to the
    configured holding period so they don't drift to whatever the LLM picks.
    """
    h = get_horizon()
    return (
        f" Frame all conclusions for a {h['label']} horizon "
        f"(typical holding period: {h['holding_period']}). "
        f"Any price targets, stop levels, and time_horizon fields must reflect "
        f"this holding period."
    )


def get_analyst_horizon_instruction() -> str:
    """Sentence appended to analyst system messages.

    Tells the analyst which lookback window to use when calling
    look-back-aware tools (get_global_news, get_news date ranges).
    """
    h = get_horizon()
    return (
        f" When tools accept a look-back window (e.g. get_global_news, "
        f"get_news date ranges), use {h['lookback_days']} days. "
        f"Frame conclusions for a {h['label']} horizon "
        f"(typical holding period: {h['holding_period']})."
    )


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
