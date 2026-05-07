"""Fork-specific prompt overlays composed onto upstream agent prompts.

Upstream's analyst and decision-agent prompts are long inline strings.
The fork has been extending those strings inline — Indian-market clauses,
anti-narrator instructions, numeric-fidelity rules, horizon and language
directives — which created a wide rebase surface: every upstream prompt
edit can collide with our additions.

This module is the seam. Each overlay is a small, named, pure function
that returns a prompt fragment. Agents compose ``upstream_base + overlay
+ overlay + ...`` so the upstream-shaped text stays intact and our
clauses move only when we change them.

Conventions:

- Overlays return ``str``. Empty string is the no-op (e.g. an
  India-specific overlay returns ``""`` for non-Indian tickers).
- Each overlay leads with its own whitespace separator so callers can
  concatenate without tracking spacing rules.
- No LLM clients, no LangGraph, no IO. Pure string composition.

Migration map — fork-specific prompt fragments that should move here as
the seam matures (track in this list, not in agent files):

- analysts/market_analyst.py: Indian F&O get_fno_oi clause
  (``india_market_analyst_overlay``).
- analysts/fundamentals_analyst.py: Indian shareholding / pledge clause
  (``india_fundamentals_overlay``).
- analysts/social_media_analyst.py: horizon/language/instrument lines
  (subsumed by ``analyst_common_overlay``).
- agents/researchers/bull_researcher.py and bear_researcher.py: numeric-
  fidelity and anti-repetition clauses (``debate_numeric_fidelity_overlay``).
- agents/risk_mgmt/*.py: risk-debate phrasing rules
  (``risk_debate_overlay``).
- agents/utils/agent_utils.py: ``build_instrument_context`` already lives
  here and should be kept until a deliberate wrap is justified.
"""

from __future__ import annotations

from tradingagents.dataflows.nse_client import is_indian_ticker


def narrator_suppression_overlay() -> str:
    """Anti-narrator clause appended to analyst system messages.

    Some providers preface the final answer with "Now I have the data..."
    or "Let me compile..." prose. Those lines get saved verbatim into
    the report file and pollute the rendered output. The overlay tells
    the model to skip the meta-narration and start directly with the
    report content.
    """
    return (
        " When you have all the data you need and are producing the final "
        "report, begin your response directly with the report content "
        "(e.g. a heading or the first analytical paragraph). Do NOT preface "
        "the report with sentences like 'Now I have all the data needed.' "
        "or 'Let me compile the analysis.' — those narrator-style intros "
        "are saved verbatim into the report file."
    )


def india_news_overlay(ticker: str) -> str:
    """India-specific news/macro tool instructions for the news analyst.

    Indian equities move heavily on NSE-filed corporate announcements
    (board meetings, results, dividends, promoter pledge changes,
    SEBI Reg 7(2) insider trades, bulk/block deals) and India-specific
    macro drivers (INR/USD, Brent, Nifty levels, India VIX, FII vs DII
    cash-market net flow) that don't surface in headline news. This
    overlay points the analyst at ``get_corporate_announcements`` and
    ``get_india_macro`` for Indian tickers, and is a no-op otherwise.
    """
    if not is_indian_ticker(ticker):
        return ""
    return (
        " For this Indian ticker, also call "
        "get_corporate_announcements(ticker, look_back_days)"
        " — this surfaces NSE-filed catalysts (board meetings, results, dividends,"
        " promoter pledge changes, insider trades under SEBI Reg 7(2), bulk/block deals)"
        " that are not in Yahoo news but routinely move Indian stocks. Also call"
        " get_india_macro() — INR/USD, Brent, Nifty levels, India VIX, and today's"
        " FII vs DII cash-market net flow. Indian equities move heavily on these"
        " numeric drivers and they're not visible in headline news."
    )
