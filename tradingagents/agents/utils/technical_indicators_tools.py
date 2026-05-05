from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.nse_fno import get_fno_oi_nse

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)


@tool
def get_fno_oi(
    ticker: Annotated[str, "Ticker symbol (e.g. 'RELIANCE.NS')"],
) -> str:
    """
    EOD F&O snapshot for an NSE-listed ticker — total Call/Put OI,
    Put-Call Ratio, max-pain strike, and the top 3 OI strikes on each side
    (which read as derivative-implied resistance/support).

    For non-Indian tickers and Indian stocks not in the F&O segment,
    returns a "not applicable" notice.

    Args:
        ticker (str): Ticker symbol (Indian listings: .NS or .BO suffix)

    Returns:
        str: Markdown F&O snapshot
    """
    return get_fno_oi_nse(ticker)