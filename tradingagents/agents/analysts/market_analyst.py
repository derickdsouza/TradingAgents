from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.agents.utils.technical_indicators_tools import get_fno_oi
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.nse_client import is_indian_ticker


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_stock_data,
            get_indicators,
        ]
        if is_indian_ticker(ticker):
            tools.append(get_fno_oi)

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.
- chandelier: Chandelier Exit (N=22, k=3 ATR) — Chuck LeBeau's ATR-based trailing stops. Returns both stop levels per bar with a signed offset from close (negative = stop below close, positive = above). `CE long X.XX (-Y.Y%)` is the trailing stop for an open long (Y.Y% drawdown buffer). `CE short X.XX (+Y.Y%)` is the trailing stop for a short — a negative offset there means close has rallied past it (short invalidated, longs in control). Usage: Quote the actual stop level when recommending swing entries — pair with breakout_20=BREAKOUT and minervini_trend=PASS. Tips: A close below CE long is the textbook exit; don't second-guess it.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.
- obv: OBV: On-Balance Volume — cumulative signed volume. Usage: Detect institutional accumulation/distribution; rising OBV with rising price confirms a trend, OBV/price divergence warns of weakness. Tips: Read as slope, not absolute. OBV breaking out before price is a leading swing-trade signal.
- mfi: MFI: Money Flow Index — momentum indicator using price and volume. Usage: Identify overbought (>80) / oversold (<20) zones and confirm reversals. Tips: Divergence vs price is a high-quality reversal signal.
- rvol_20: Relative Volume vs 20-day average. Usage: Confirm institutional participation in a breakout — RVOL >= 2 with a price breakout is the canonical institutional swing signal. Tips: RVOL < 1 on a breakout is a red flag (no follow-through).
- vsa: VSA (Volume Spread Analysis, Tom Williams) — classifies each bar by spread × volume × close-location. Returns: STRENGTH, WEAKNESS, NO-DEMAND, NO-SUPPLY, STOPPING-VOLUME, BUYING-CLIMAX, CLIMAX-ABSORPTION, SPRING (bullish false break of support on volume), UPTHRUST (bearish false break of resistance on volume), or 'neutral'. Usage: Reads supply/demand intent — answers 'who is in control?'. Tips: SPRING + next-day BREAKOUT = high-conviction long; NO-DEMAND in an uptrend warns of reversal.

Minervini SEPA Indicators (institutional swing setups):
- minervini_trend: Mark Minervini's Trend Template — 7-of-8 condition checklist for confirmed uptrends (price > 50/150/200 SMA, 50 > 150 > 200 SMA stack, 200-SMA rising, price within 25% of 52-week high and 30%+ above 52-week low). Returns 'PASS (N/7)' or 'FAIL (N/7): missing X'. Usage: Use as a hard go/no-go filter — Minervini won't trade longs unless this passes. Tips: 8th criterion (RS rating ≥ 70) is NOT computed here because it's market-relative; layer that in qualitatively from sentiment/news context.
- pocket_pivot: Pocket Pivot signal — close >= highest down-day volume of the last 10 days, on an up day with above-average volume. Returns 'POCKET-PIVOT' or 'neutral'. Usage: Minervini's early-entry signal for stocks in stage-2 uptrends, often appearing before a full breakout. Tips: Best inside an existing minervini_trend=PASS regime. A Pocket Pivot from a tight base is the textbook Minervini swing entry.
- vcp: Volatility Contraction Pattern — detects successive shallower pullbacks (e.g. 25%→15%→8%) within a 60-day base with volume drying up. Returns 'VCP-FORMING', 'VCP-COMPLETE', 'VCP-BREAKOUT', or 'no-VCP', with the contraction depths quoted. Usage: The Minervini gold-standard base structure for swing entries — VCP-COMPLETE + breakout = textbook setup. Tips: Pair with minervini_trend=PASS and rising RS for highest conviction.
- tight_3w: 3-Week Tight pattern — last 3 weekly closes within ~1% of each other. Returns '3WT (...)' or 'neutral'. Usage: Reads as institutional accumulation in a flat zone; a pocket-pivot or breakout out of a 3WT is one of Minervini's highest-quality buy signals. Tips: Often appears inside a VCP-COMPLETE base.

Trend-Strength / Quality Indicators:
- adx: ADX: Average Directional Index, 0-100. Usage: ADX > 25 = trending market (favor breakouts/trend-following); ADX < 20 = range-bound (favor mean reversion). Tips: A rising ADX above 25 with a price breakout is a high-quality swing setup.
- pdi: +DI (Plus Directional Indicator, 14) — the bullish half of Wilder's DMI. Usage: Read with adx + ndi to give ADX a direction. +DI > -DI = bullish bias; the wider the spread, the cleaner the trend. Tips: A +DI/-DI bullish crossover with ADX rising above 25 is Wilder's textbook long entry. Avoid acting in low-ADX (<20) regimes — high +DI there is just noise.
- ndi: -DI (Minus Directional Indicator, 14) — the bearish half of Wilder's DMI. Usage: -DI > +DI = bearish bias / exit-long signal. Tips: -DI rising while price still grinds higher = divergence warning that the up-leg is losing breadth. Confirm with ADX > 20-25 before acting on a -DI > +DI cross.
- aroon_25: Aroon Up / Down (25) — Tushar Chande's trend-FRESHNESS indicator. Returns `Aroon-Up: X | Aroon-Down: Y | spread: ±Z (TAG)` where TAG ∈ {BULLISH-FRESH, BULLISH, NEUTRAL, BEARISH, BEARISH-FRESH}. Usage: Distinct from ADX — measures how recently new highs/lows printed. The high-conviction warning signal: rising price + falling Aroon-Up = aging trend losing its ability to make new highs. Tips: BULLISH-FRESH + breakout_20=BREAKOUT is a textbook entry; Aroon-Up dropping below 50 while price is still rising is one of the highest-quality early-exit signals available.
- guppy: Guppy MMA (GMMA) — short EMA ribbon (3,5,8,10,12,15) vs long EMA ribbon (30,35,40,45,50,60). Returns a regime label: BULLISH-FAN, BULLISH-COMPRESS, BEARISH-FAN, BEARISH-COMPRESS, or TRANSITION. Usage: Top-tier trend-quality filter — BULLISH-FAN + breakout_20=BREAKOUT is a textbook swing entry. Tips: TRANSITION often precedes either continuation or reversal — disambiguate with OBV slope.

Breakout Trigger:
- breakout_20: 20-day volume-confirmed breakout flag. Returns 'BREAKOUT' when close > prior 20-day high AND RVOL_20 >= 1.5; 'BREAKDOWN' for the inverse; otherwise reports distance to range edges. Usage: Direct swing-trade trigger — confirm with ADX > 25, BULLISH-FAN guppy, and rising OBV.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions.

For SHORT-HORIZON / SWING setups specifically: prioritise the institutional-volume toolkit — combine `breakout_20`, `rvol_20`, `adx`, and `obv` with one trend filter (e.g. `close_50_sma` or `close_10_ema`). A 'BREAKOUT' from `breakout_20` confirmed by ADX > 25 and rising OBV is the highest-conviction swing entry. Always quote the actual `breakout_20` and `rvol_20` values rather than just describing them qualitatively."""
            + ((
                " For this Indian ticker, also call get_fno_oi(ticker) — "
                "it returns the EOD F&O snapshot (total Call/Put OI, "
                "Put-Call Ratio, max-pain strike, and top OI strikes which read "
                "as derivative-implied resistance/support). Quote PCR and "
                "max-pain when forming the swing thesis; PCR < 0.7 is "
                "call-heavy/bullish-positioned, > 1.3 is put-heavy."
            ) if is_indian_ticker(ticker) else "")
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + " When you have all the data you need and are producing the final report, begin your response directly with the report content (e.g. a heading or the first analytical paragraph). Do NOT preface the report with sentences like 'Now I have all the data needed.' or 'Let me compile the analysis.' — those narrator-style intros are saved verbatim into the report file."
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
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
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
            "market_report": report,
        }

    return market_analyst_node
