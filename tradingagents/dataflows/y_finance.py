from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import os
from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, load_ohlcv, filter_financials_by_date

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
        "obv": (
            "OBV: On-Balance Volume is a cumulative volume flow indicator that adds volume on up days and subtracts it on down days. "
            "Usage: Detect accumulation/distribution by institutions; rising OBV with rising price confirms a trend, while OBV/price divergence warns of weakness. "
            "Tips: Best read as a slope/trend, not an absolute number. Watch for OBV breaking out before price as a leading swing-trade signal."
        ),
        "adx": (
            "ADX: Average Directional Index measures trend strength (not direction) on a 0-100 scale. "
            "Usage: ADX > 25 indicates a trending market suitable for trend-following / breakout entries; ADX < 20 indicates a range-bound market favoring mean-reversion. "
            "Tips: Combine with +DI/-DI for direction. A rising ADX above 25 alongside a price breakout is a high-quality swing setup."
        ),
        "pdi": (
            "+DI (Plus Directional Indicator, 14): The bullish half of the Wilder DMI system. Measures the share of recent price movement attributable to up-days, on a 0-100 scale. "
            "Usage: Read alongside ADX and -DI to give ADX a direction. +DI > -DI = bullish bias; the wider the spread, the more one-sided the trend. "
            "Tips: A +DI/-DI crossover (where +DI crosses above -DI) while ADX is rising above 25 is the canonical Wilder long-entry trigger. Useless without ADX context — a high +DI in a low-ADX (ADX<20) regime just means a noisy uptrend with no follow-through."
        ),
        "ndi": (
            "-DI (Minus Directional Indicator, 14): The bearish half of the Wilder DMI system. Measures the share of recent price movement attributable to down-days, on a 0-100 scale. "
            "Usage: Read alongside ADX and +DI. -DI > +DI = bearish bias; a -DI cross above +DI with rising ADX is the canonical Wilder short-entry / exit-long signal. "
            "Tips: When -DI is rising while price is still grinding higher, treat it as a divergence warning that the up-leg is losing breadth. Only act on -DI > +DI when ADX confirms (>20-25) — otherwise it's just chop."
        ),
        "rvol_20": (
            "RVOL (20): Relative Volume vs 20-day average volume. Values >= 1.5 mean today is at least 1.5x the average. "
            "Usage: Confirm institutional participation in a breakout or reversal. RVOL >= 2 alongside a price breakout is the canonical institutional swing signal. "
            "Tips: RVOL < 1 on a breakout is a red flag (no follow-through). Always pair with a breakout / pattern signal, not used alone."
        ),
        "breakout_20": (
            "Breakout (20-day, volume-confirmed): Returns 'BREAKOUT' when today's close > prior 20-day high AND RVOL_20 >= 1.5; "
            "'BREAKDOWN' when close < prior 20-day low AND RVOL_20 >= 1.5; flags weak-volume range breaks separately; otherwise reports distance to range edges. "
            "Usage: A direct swing-trade trigger. BREAKOUT with strong RVOL is the institutional accumulation pattern. "
            "Tips: Always check ADX > 25 and OBV trend for confirmation. Breakouts in low-ADX (range) markets often fail."
        ),
        "guppy": (
            "Guppy MMA (GMMA): Two ribbons of EMAs — a short-term 'trader' group (3,5,8,10,12,15) and a long-term 'investor' group (30,35,40,45,50,60). "
            "Returns a regime label: BULLISH-FAN (trader group above investor group and expanding = strongest uptrend), "
            "BULLISH-COMPRESS (above but contracting = trend losing momentum), BEARISH-FAN / BEARISH-COMPRESS (mirror), or TRANSITION (groups overlapping = trend change in progress). "
            "Usage: Top-tier trend-quality filter. BULLISH-FAN with breakout_20=BREAKOUT is a textbook swing entry. "
            "Tips: TRANSITION + BULLISH-COMPRESS often precedes either a continuation rally or a reversal — pair with OBV slope to disambiguate."
        ),
        "vsa": (
            "VSA: Volume Spread Analysis (Tom Williams) — classifies each bar by the relationship between price spread, volume, and close location. "
            "Returns one of: STRENGTH, WEAKNESS, NO-DEMAND, NO-SUPPLY, STOPPING-VOLUME, BUYING-CLIMAX, CLIMAX-ABSORPTION, SPRING (bullish false break of support on volume), UPTHRUST (bearish false break of resistance on volume), or 'neutral'. "
            "Usage: Reads supply/demand intent at the bar level — answers 'who is in control?' rather than just 'where is price?'. "
            "Tips: SPRING + breakout_20=BREAKOUT next day = high-conviction long. UPTHRUST + BEARISH-FAN guppy = high-conviction short. NO-DEMAND in an uptrend warns of impending reversal."
        ),
        "minervini_trend": (
            "Minervini Trend Template (SEPA): Mark Minervini's full 8-condition checklist for stocks in confirmed Stage-2 uptrends. "
            "Conditions: price > 150 & 200 SMA; 150 > 200 SMA; 200-SMA rising for ~1 month; 50 > 150 & 200 SMA; price > 50 SMA; price ≥ 30% above 52w low; price within 25% of 52w high; AND RS line rising vs benchmark (Nifty 500 for .NS/.BO, SPY for US, etc.). "
            "Returns 'PASS (N/8)' or 'FAIL (N/8): missing <list>'. Falls back to 7 conditions if the benchmark fetch fails. "
            "Usage: Hard go/no-go filter — Minervini won't take longs unless this passes. "
            "Tips: A PASS with VCP-COMPLETE base + pocket_pivot is the textbook SEPA swing entry. Requires ~1 year of price history."
        ),
        "pocket_pivot": (
            "Pocket Pivot (Minervini / O'Neil): An up day where today's volume exceeds the highest down-day volume of the prior 10 sessions, AND volume is above the 50-day average. "
            "Returns 'POCKET-PIVOT (...)' or 'neutral'. "
            "Usage: Early-entry signal for stocks already in an uptrend; often precedes the more obvious 20-day breakout. "
            "Tips: Highest conviction when minervini_trend=PASS, the stock is in a tight base, and OBV is rising. A pocket pivot from a 3-week tight pattern is the textbook SEPA swing entry."
        ),
        "vcp": (
            "VCP (Volatility Contraction Pattern, Minervini): Successive shallower pullbacks within a 60-day base, "
            "with volume drying up into the apex — the canonical Stage-2 base structure. "
            "Returns 'VCP-BREAKOUT' (close above pivot high on >1.5x avg volume), 'VCP-COMPLETE' (final contraction <8% with volume dry-up), "
            "'VCP-FORMING' (contractions shallowing but not yet tight), or 'no-VCP'. "
            "Usage: The Minervini gold standard for swing entries — buy the breakout from a VCP-COMPLETE base with rising RS line. "
            "Tips: Pair with minervini_trend=PASS and pocket_pivot or breakout_20=BREAKOUT for confirmation. Best when contractions show 25-50% then 15-25% then <10%."
        ),
        "tight_3w": (
            "3-Week Tight (Minervini): Three consecutive weekly closes within ~1% of each other. "
            "Returns '3WT (...)' or 'neutral'. "
            "Usage: Signals institutional accumulation after a prior advance — buyers are quietly absorbing supply at flat closes. "
            "Tips: A pocket pivot or breakout out of a 3WT is one of Minervini's highest-quality entries. Almost always appears inside a VCP-COMPLETE base."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


_CUSTOM_INDICATORS = {
    "obv", "rvol_20", "breakout_20", "guppy", "vsa",
    "minervini_trend", "pocket_pivot", "vcp", "tight_3w",
}

# Guppy Multiple Moving Average (GMMA) groups.
_GUPPY_SHORT_PERIODS = (3, 5, 8, 10, 12, 15)
_GUPPY_LONG_PERIODS = (30, 35, 40, 45, 50, 60)


def _resolve_benchmark(symbol: str) -> str:
    """Pick the right equity benchmark for a Minervini RS comparison.

    Indian listings benchmark to Nifty 500 (the broadest liquid index that
    matches IBD's universe-wide RS spirit), other suffixes to their
    home-market major, everything else to SPY.
    """
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "^CRSLDX"  # Nifty 500
    if s.endswith(".L"):
        return "^FTSE"
    if s.endswith(".HK"):
        return "^HSI"
    if s.endswith(".T"):
        return "^N225"
    if s.endswith(".TO"):
        return "^GSPTSE"
    if s.endswith(".AX"):
        return "^AXJO"
    return "SPY"


def _load_benchmark_close(symbol: str, curr_date: str) -> "pd.Series | None":
    """Load the benchmark close series indexed by YYYY-MM-DD strings.

    Returns None if the benchmark fetch fails so the caller can degrade
    gracefully (report 7/7 instead of 8/8).
    """
    from stockstats import wrap

    try:
        bench_data = load_ohlcv(_resolve_benchmark(symbol), curr_date)
    except Exception:
        return None
    bench_df = wrap(bench_data)
    bench_df["Date"] = bench_df["Date"].dt.strftime("%Y-%m-%d")
    return bench_df.set_index("Date")["close"]


def _compute_custom_indicator(
    df: pd.DataFrame,
    indicator: str,
    symbol: str | None = None,
    curr_date: str | None = None,
) -> pd.Series:
    """Compute indicators that stockstats doesn't support natively.

    Operates on a stockstats-wrapped frame whose columns are lowercase
    OHLCV. Returns a Series aligned to df.index.
    """
    if indicator == "obv":
        # On-Balance Volume: cumulative signed volume.
        direction = df["close"].diff().fillna(0)
        signed = df["volume"].where(direction > 0, -df["volume"]).where(direction != 0, 0)
        return signed.cumsum()

    if indicator == "rvol_20":
        return df["volume"] / df["volume"].rolling(20).mean()

    if indicator == "breakout_20":
        prior_high = df["high"].rolling(20).max().shift(1)
        prior_low = df["low"].rolling(20).min().shift(1)
        rvol = df["volume"] / df["volume"].rolling(20).mean()
        out = []
        for close, ph, pl, rv in zip(df["close"], prior_high, prior_low, rvol):
            if pd.isna(ph) or pd.isna(pl) or pd.isna(rv):
                out.append("N/A")
                continue
            if close > ph and rv >= 1.5:
                out.append(f"BREAKOUT (close {close:.2f} > 20d high {ph:.2f}, RVOL {rv:.2f}x)")
            elif close < pl and rv >= 1.5:
                out.append(f"BREAKDOWN (close {close:.2f} < 20d low {pl:.2f}, RVOL {rv:.2f}x)")
            elif close > ph:
                out.append(
                    f"above 20d high {ph:.2f} on weak volume (RVOL {rv:.2f}x — no institutional confirmation)"
                )
            elif close < pl:
                out.append(
                    f"below 20d low {pl:.2f} on weak volume (RVOL {rv:.2f}x — no institutional confirmation)"
                )
            else:
                pct_to_high = (ph - close) / close * 100
                pct_to_low = (close - pl) / close * 100
                out.append(
                    f"in-range (RVOL {rv:.2f}x; {pct_to_high:.1f}% to 20d high, {pct_to_low:.1f}% to 20d low)"
                )
        return pd.Series(out, index=df.index)

    if indicator == "vsa":
        # Volume Spread Analysis — Tom Williams bar-by-bar classification.
        # Tags each bar with the dominant VSA signal so the LLM can read
        # supply/demand intent rather than just price.
        spread = df["high"] - df["low"]
        spread_avg = spread.rolling(20).mean()
        vol_avg = df["volume"].rolling(20).mean()
        prior_high = df["high"].shift(1)
        prior_low = df["low"].shift(1)

        out = []
        for i, (o, h, l, c, v, sp, sp_a, v_a, p_h, p_l) in enumerate(zip(
            df["open"], df["high"], df["low"], df["close"], df["volume"],
            spread, spread_avg, vol_avg, prior_high, prior_low,
        )):
            if pd.isna(sp_a) or pd.isna(v_a) or sp_a == 0 or v_a == 0:
                out.append("N/A")
                continue

            # Spread bucket
            if sp > 1.5 * sp_a:
                spread_cat = "wide"
            elif sp < 0.7 * sp_a:
                spread_cat = "narrow"
            else:
                spread_cat = "normal"

            # Volume bucket
            if v > 2.0 * v_a:
                vol_cat = "ultra"
            elif v > 1.5 * v_a:
                vol_cat = "high"
            elif v < 0.7 * v_a:
                vol_cat = "low"
            else:
                vol_cat = "normal"

            up_bar = c > o
            close_loc = (c - l) / sp if sp > 0 else 0.5  # 0=at low, 1=at high

            # Spring / Upthrust (false break + recovery, on volume)
            if not pd.isna(p_l) and l < p_l and c > p_l and vol_cat in ("high", "ultra"):
                out.append(f"SPRING (false break of prior low {p_l:.2f}, closes back inside on {vol_cat} volume — bullish reversal)")
                continue
            if not pd.isna(p_h) and h > p_h and c < p_h and vol_cat in ("high", "ultra"):
                out.append(f"UPTHRUST (false break of prior high {p_h:.2f}, closes back inside on {vol_cat} volume — bearish reversal)")
                continue

            # Wide-spread strength / weakness
            if up_bar and spread_cat == "wide" and vol_cat in ("high", "ultra") and close_loc > 0.7:
                out.append(f"STRENGTH (wide-up bar on {vol_cat} volume, close {close_loc*100:.0f}% up the bar — demand)")
                continue
            if not up_bar and spread_cat == "wide" and vol_cat in ("high", "ultra") and close_loc < 0.3:
                out.append(f"WEAKNESS (wide-down bar on {vol_cat} volume, close {close_loc*100:.0f}% up the bar — supply)")
                continue

            # No-demand / No-supply (low-vol narrow bars)
            if up_bar and spread_cat == "narrow" and vol_cat == "low":
                out.append("NO-DEMAND (narrow up bar on low volume — bearish in uptrend, neutral in range)")
                continue
            if not up_bar and spread_cat == "narrow" and vol_cat == "low":
                out.append("NO-SUPPLY (narrow down bar on low volume — bullish in downtrend, neutral in range)")
                continue

            # Stopping volume / Buying climax / Selling climax
            if not up_bar and vol_cat == "ultra" and close_loc > 0.5:
                out.append("STOPPING-VOLUME (down bar on ultra volume, closes off the lows — bullish reversal signal)")
                continue
            if up_bar and vol_cat == "ultra" and close_loc < 0.5:
                out.append("BUYING-CLIMAX (up bar on ultra volume, closes off the highs — bearish reversal signal)")
                continue
            if vol_cat == "ultra" and 0.4 <= close_loc <= 0.6:
                out.append("CLIMAX-ABSORPTION (ultra volume, close mid-range — supply meeting demand)")
                continue

            out.append("neutral")
        return pd.Series(out, index=df.index)

    if indicator == "guppy":
        # Guppy Multiple Moving Average — six short EMAs (trader group) vs six
        # long EMAs (investor group). Encodes the regime as a single label so
        # the LLM doesn't have to reconcile twelve series.
        short_emas = pd.concat(
            [df["close"].ewm(span=p, adjust=False).mean() for p in _GUPPY_SHORT_PERIODS],
            axis=1,
        )
        long_emas = pd.concat(
            [df["close"].ewm(span=p, adjust=False).mean() for p in _GUPPY_LONG_PERIODS],
            axis=1,
        )
        short_min = short_emas.min(axis=1)
        short_max = short_emas.max(axis=1)
        long_min = long_emas.min(axis=1)
        long_max = long_emas.max(axis=1)
        short_spread = short_max - short_min
        spread_avg = short_spread.rolling(20).mean()

        out = []
        for s_min, s_max, l_min, l_max, s_spread, s_avg in zip(
            short_min, short_max, long_min, long_max, short_spread, spread_avg
        ):
            if pd.isna(s_min) or pd.isna(l_max) or pd.isna(s_avg) or s_avg == 0:
                out.append("N/A")
                continue
            fanning = s_spread > s_avg  # short group expanding = strong trend
            if s_min > l_max:
                regime = "BULLISH-FAN" if fanning else "BULLISH-COMPRESS"
            elif s_max < l_min:
                regime = "BEARISH-FAN" if fanning else "BEARISH-COMPRESS"
            else:
                regime = "TRANSITION"
            out.append(
                f"{regime} (short EMAs {s_min:.2f}-{s_max:.2f}, long EMAs {l_min:.2f}-{l_max:.2f})"
            )
        return pd.Series(out, index=df.index)

    if indicator == "minervini_trend":
        # Minervini Trend Template — full 8 SEPA conditions.
        # The 8th (RS rating) compares stock vs benchmark price action: we
        # proxy IBD's universe-wide RS rating with a stock/benchmark ratio
        # whose 50-day SMA is rising (RS line trending up = outperforming).
        # Benchmark auto-selects: Nifty 500 for .NS/.BO, SPY for US, etc.
        sma_50 = df["close"].rolling(50).mean()
        sma_150 = df["close"].rolling(150).mean()
        sma_200 = df["close"].rolling(200).mean()
        sma_200_22d_ago = sma_200.shift(22)  # ~1 month back
        high_52w = df["high"].rolling(252).max()
        low_52w = df["low"].rolling(252).min()

        # 8th condition: RS line vs benchmark, slope-up filter.
        rs_line = None
        rs_line_sma = None
        rs_line_sma_prev = None
        bench_label = None
        if symbol and curr_date:
            bench_close = _load_benchmark_close(symbol, curr_date)
            if bench_close is not None:
                df_dates = df["Date"] if "Date" in df.columns else df.index
                # Align benchmark to stock's date index by reindex+ffill.
                bench_aligned = bench_close.reindex(df_dates).ffill()
                rs_line = (df["close"].values / bench_aligned.values)
                rs_line = pd.Series(rs_line, index=df.index)
                rs_line_sma = rs_line.rolling(50).mean()
                rs_line_sma_prev = rs_line_sma.shift(22)
                bench_label = _resolve_benchmark(symbol)

        out = []
        for i, (c, s50, s150, s200, s200_prev, h52, l52) in enumerate(zip(
            df["close"], sma_50, sma_150, sma_200, sma_200_22d_ago, high_52w, low_52w
        )):
            if any(pd.isna(x) for x in (c, s50, s150, s200, s200_prev, h52, l52)):
                out.append("N/A (insufficient history — needs ~1 year of data)")
                continue

            checks = [
                ("price > 150-SMA & 200-SMA", c > s150 and c > s200),
                ("150-SMA > 200-SMA", s150 > s200),
                ("200-SMA rising (1 month)", s200 > s200_prev),
                ("50-SMA > 150-SMA & 200-SMA", s50 > s150 and s50 > s200),
                ("price > 50-SMA", c > s50),
                ("price >= 30% above 52w low", c >= 1.30 * l52),
                ("price within 25% of 52w high", c >= 0.75 * h52),
            ]

            # 8th: RS line above its 50-SMA AND that SMA rising = outperforming
            rs_note = ""
            if rs_line is not None:
                rs_val = rs_line.iloc[i]
                rs_sma_val = rs_line_sma.iloc[i]
                rs_sma_prev_val = rs_line_sma_prev.iloc[i]
                if any(pd.isna(x) for x in (rs_val, rs_sma_val, rs_sma_prev_val)):
                    out.append("N/A (insufficient RS history)")
                    continue
                rs_ok = rs_val > rs_sma_val and rs_sma_val > rs_sma_prev_val
                checks.append((f"RS line rising vs {bench_label}", rs_ok))
            else:
                rs_note = " (RS skipped — benchmark fetch failed)"

            total = len(checks)
            passed = sum(1 for _, ok in checks if ok)
            failed = [name for name, ok in checks if not ok]
            verdict = "PASS" if passed == total else "FAIL"
            if failed:
                out.append(f"{verdict} ({passed}/{total}): missing {'; '.join(failed)}{rs_note}")
            else:
                out.append(f"{verdict} ({passed}/{total}){rs_note}")
        return pd.Series(out, index=df.index)

    if indicator == "pocket_pivot":
        # Pocket Pivot (Minervini / O'Neil): an up day where today's volume
        # exceeds the highest down-day volume of the prior 10 trading days.
        out = []
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        vol_avg_50 = df["volume"].rolling(50).mean().values

        for i in range(len(df)):
            if i < 10 or pd.isna(vol_avg_50[i]):
                out.append("N/A (insufficient history)")
                continue
            up_today = closes[i] > opens[i]
            # Highest down-day volume in the prior 10 days (i-10 .. i-1).
            window_down_vols = [
                volumes[j] for j in range(i - 10, i) if closes[j] < opens[j]
            ]
            max_down_vol = max(window_down_vols) if window_down_vols else 0
            above_avg = volumes[i] > vol_avg_50[i]
            if up_today and volumes[i] > max_down_vol and above_avg:
                out.append(
                    f"POCKET-PIVOT (vol {int(volumes[i])} > max down-day vol {int(max_down_vol)}, "
                    f"and > 50d avg {int(vol_avg_50[i])})"
                )
            else:
                out.append("neutral")
        return pd.Series(out, index=df.index)

    if indicator == "vcp":
        # Volatility Contraction Pattern (Minervini): successive base
        # contractions where each pullback is shallower than the previous,
        # and volume dries up into the apex. Walks back from each bar and
        # finds 2-3 contractions in a ~12-week window, then classifies.
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vol_avg_50 = df["volume"].rolling(50).mean().values

        # Identify rolling local pivots (5-bar fractals) — local highs/lows.
        n = len(df)
        pivot_high = [False] * n
        pivot_low = [False] * n
        for i in range(2, n - 2):
            if (
                highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2]
                and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]
            ):
                pivot_high[i] = True
            if (
                lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2]
                and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]
            ):
                pivot_low[i] = True

        out = []
        for i in range(n):
            if i < 60 or pd.isna(vol_avg_50[i]):
                out.append("N/A (insufficient history)")
                continue

            # Walk back ~12 weeks (60 trading days) and collect alternating
            # high→low contraction depths.
            window_start = max(0, i - 60)
            contractions = []
            j = i
            # Find last pivot high at or before i.
            last_high_idx = None
            for k in range(i, window_start - 1, -1):
                if pivot_high[k]:
                    last_high_idx = k
                    break
            cursor = last_high_idx
            while cursor is not None and cursor > window_start and len(contractions) < 4:
                # Find next pivot low before cursor.
                low_idx = None
                for k in range(cursor - 1, window_start - 1, -1):
                    if pivot_low[k]:
                        low_idx = k
                        break
                if low_idx is None:
                    break
                # Find next pivot high before low_idx.
                prev_high_idx = None
                for k in range(low_idx - 1, window_start - 1, -1):
                    if pivot_high[k]:
                        prev_high_idx = k
                        break
                if prev_high_idx is None:
                    break
                depth_pct = (highs[prev_high_idx] - lows[low_idx]) / highs[prev_high_idx] * 100
                contractions.append(float(depth_pct))
                cursor = prev_high_idx

            # Most recent contraction first; reverse so chronological.
            contractions = list(reversed(contractions))

            if len(contractions) < 2:
                out.append("no-VCP (insufficient pivots in 60d window)")
                continue

            # Test successive shallowing (allow tiny tolerance).
            shallowing = all(
                contractions[k + 1] < contractions[k] * 0.95
                for k in range(len(contractions) - 1)
            )
            latest_depth = contractions[-1]

            # Volume dry-up: last 5-day avg vs prior 20-day avg.
            recent_vol = volumes[max(0, i - 4): i + 1].mean()
            prior_vol = volumes[max(0, i - 25): max(0, i - 5)].mean() if i >= 5 else 0
            vol_dryup = prior_vol > 0 and recent_vol < 0.85 * prior_vol

            if not shallowing:
                out.append(
                    f"no-VCP (contractions {[round(c, 1) for c in contractions]}% — not shallowing)"
                )
                continue

            # Distance from current close to latest pivot high.
            if last_high_idx is not None:
                pivot_price = highs[last_high_idx]
                dist_pct = (pivot_price - closes[i]) / pivot_price * 100
                if closes[i] > pivot_price and volumes[i] > 1.5 * vol_avg_50[i]:
                    label = "VCP-BREAKOUT"
                elif latest_depth < 8 and vol_dryup:
                    label = "VCP-COMPLETE"
                else:
                    label = "VCP-FORMING"
                out.append(
                    f"{label} (contractions {[round(c, 1) for c in contractions]}%, "
                    f"latest {latest_depth:.1f}%, {dist_pct:.1f}% below pivot {pivot_price:.2f}, "
                    f"vol-dryup={vol_dryup})"
                )
            else:
                out.append("no-VCP (no anchor pivot high)")
        return pd.Series(out, index=df.index)

    if indicator == "tight_3w":
        # 3-Week Tight (Minervini): three consecutive weekly closes within
        # ~1% of each other = institutional accumulation in a tight range.
        # We compute per trading day by looking at the trailing 15 sessions
        # (~3 weeks) and checking weekly-close dispersion.
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"])
        else:
            dates = pd.to_datetime(df.index)

        # Map each row to its ISO year-week so we can pick weekly closes.
        iso = dates.dt.isocalendar() if hasattr(dates, "dt") else pd.DataFrame(
            {"year": dates.year, "week": dates.isocalendar().week}
        )
        year_week = list(zip(iso["year"], iso["week"]))

        out = []
        for i in range(len(df)):
            if i < 15:
                out.append("N/A (insufficient history)")
                continue
            # Walk back, taking the LAST close per week, until we have 3.
            seen_weeks = []
            weekly_closes = []
            for k in range(i, max(-1, i - 25), -1):
                yw = year_week[k]
                if yw in seen_weeks:
                    continue
                seen_weeks.append(yw)
                weekly_closes.append(float(df["close"].iloc[k]))
                if len(weekly_closes) == 3:
                    break
            if len(weekly_closes) < 3:
                out.append("neutral (insufficient weeks)")
                continue
            mean_close = sum(weekly_closes) / 3
            spread_pct = (max(weekly_closes) - min(weekly_closes)) / mean_close * 100
            if spread_pct <= 1.0:
                out.append(
                    f"3WT (weekly closes {[round(c, 2) for c in reversed(weekly_closes)]} "
                    f"within {spread_pct:.2f}% — tight base, accumulation)"
                )
            else:
                out.append(f"neutral (3w spread {spread_pct:.2f}%)")
        return pd.Series(out, index=df.index)

    raise ValueError(f"Unknown custom indicator: {indicator}")


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    if indicator in _CUSTOM_INDICATORS:
        df[indicator] = _compute_custom_indicator(
            df, indicator, symbol=symbol, curr_date=curr_date
        )
    else:
        # Triggers stockstats to calculate the indicator for all rows at once.
        df[indicator]

    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"