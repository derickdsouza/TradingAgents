from langchain_core.tools import tool
from typing import Annotated, Optional
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.nse_announcements import get_corporate_announcements_nse
from tradingagents.dataflows.india_macro import get_india_macro as _get_india_macro

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[Optional[int], "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[Optional[int], "Max articles to return; omit to use the configured default"] = None,
    ticker: Annotated[
        str,
        "Optional ticker (e.g. 'TCS.NS', 'AAPL'). Pass the instrument under "
        "analysis so the search uses region-aware macro queries — Indian "
        "tickers (.NS/.BO) trigger RBI/CPI/FII-DII/INR queries instead of "
        "the default US-Fed/CPI set.",
    ] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config
        ticker (str): Optional ticker for region-aware queries
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit, ticker=ticker)

@tool
def get_corporate_announcements(
    ticker: Annotated[str, "Ticker symbol (e.g. 'TCS.NS', 'RELIANCE.BO')"],
    look_back_days: Annotated[int, "Look-back window in days"] = 30,
    limit: Annotated[int, "Max announcements to return"] = 25,
) -> str:
    """
    Retrieve SEBI-mandated corporate announcements filed with NSE for an
    Indian-listed ticker — board meetings, results, dividends, promoter
    pledge changes, insider transactions (Reg 7(2)), bulk/block deals.
    These are catalysts that move Indian stocks but are not in Yahoo news.

    For non-Indian tickers (no .NS/.BO suffix) returns a "not applicable"
    notice — call only when analyzing Indian listings.

    Args:
        ticker (str): Ticker symbol (Indian listings: .NS or .BO suffix)
        look_back_days (int): Look-back window in days (default 30)
        limit (int): Max announcements to return (default 25)

    Returns:
        str: Markdown-formatted list of announcements
    """
    return get_corporate_announcements_nse(ticker, look_back_days, limit)


@tool
def get_india_macro(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format (informational)"] = None,
) -> str:
    """
    Snapshot of India macro-context drivers: INR/USD, Brent crude, Nifty 50,
    Nifty Bank, India VIX, and the most-recent EOD FII vs DII cash-market
    net flow. These are the numeric factors that dominate Indian equity
    price action and are not visible in headline-style news.

    Use only when analyzing Indian-listed (.NS/.BO) tickers — for non-Indian
    tickers the global macro is already covered by get_global_news.

    Args:
        curr_date (str): Current date (informational; data is live EOD)

    Returns:
        str: Markdown summary table + institutional flow
    """
    return _get_india_macro(curr_date)


@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)
