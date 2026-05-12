"""Tests for the sparse-data confidence floor on sentiment analyst output (y8l).

The sentiment analyst emits prose only — there is no structured schema in the
upstream design. To make sparse-data cases legible to downstream consumers
(Research Manager scorecard, Portfolio Manager) without rewriting the analyst
into structured output, we add two surfaces:

1. A typed ``SentimentReport`` schema with a model validator that REJECTS
   high/medium confidence claims on evidence_count ≤ 1. Code, not the LLM,
   answers the deterministic question "is this sparse?".
2. A deterministic helper that counts the visible evidence items across the
   three pre-fetched blocks (news, StockTwits, Reddit). The analyst node
   prepends a ``**Sentiment Confidence**`` banner to its prose output when
   the count is sparse so the reader's mental model is calibrated up front.

The schema validator is the regression anchor: a future analyst refactor that
moves to structured output cannot accidentally re-introduce the qwen failure
mode (one news article → high-confidence BUY).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSentimentReportSchemaConfidenceFloor:
    """y8l — the schema rejects HIGH/MEDIUM confidence on sparse evidence.
    Sparse is defined as evidence_count ≤ 1 (one article and no socials, or
    nothing at all). Below that floor the only defensible confidence value
    is LOW.
    """

    def test_one_article_high_confidence_buy_is_rejected(self):
        from tradingagents.agents.schemas import SentimentReport

        with pytest.raises(Exception):
            SentimentReport(
                final_recommendation="BUY",
                confidence="high",
                evidence_count=1,
                rationale="One bullish headline.",
            )

    def test_one_article_medium_confidence_is_rejected(self):
        from tradingagents.agents.schemas import SentimentReport

        with pytest.raises(Exception):
            SentimentReport(
                final_recommendation="BUY",
                confidence="medium",
                evidence_count=1,
                rationale="One bullish headline.",
            )

    def test_zero_articles_high_confidence_is_rejected(self):
        from tradingagents.agents.schemas import SentimentReport

        with pytest.raises(Exception):
            SentimentReport(
                final_recommendation="HOLD",
                confidence="high",
                evidence_count=0,
                rationale="No evidence at all.",
            )

    def test_one_article_low_confidence_is_accepted(self):
        from tradingagents.agents.schemas import SentimentReport

        report = SentimentReport(
            final_recommendation="BUY",
            confidence="low",
            evidence_count=1,
            rationale="One bullish headline — directional read but thin.",
        )
        assert report.confidence == "low"
        assert report.evidence_count == 1

    def test_two_articles_medium_confidence_is_accepted(self):
        """Above the floor the validator is silent — confidence is the
        model's judgement call, the schema only catches the egregious case."""
        from tradingagents.agents.schemas import SentimentReport

        report = SentimentReport(
            final_recommendation="BUY",
            confidence="medium",
            evidence_count=2,
            rationale="Two news items align bullishly.",
        )
        assert report.confidence == "medium"

    def test_ten_articles_high_confidence_is_accepted(self):
        from tradingagents.agents.schemas import SentimentReport

        report = SentimentReport(
            final_recommendation="SELL",
            confidence="high",
            evidence_count=12,
            rationale="Bearish across 12 sources.",
        )
        assert report.confidence == "high"


@pytest.mark.unit
class TestCountSentimentEvidence:
    """The counter is deterministic — the LLM does not see this number, the
    analyst does. Failure modes traced from real data:
      - The qwen failure case: one news bullet + StockTwits "<no messages
        found>" + Reddit "no posts found" in all three subreddits → 1 item.
      - Truly silent: news block "No news headlines available" + both
        socials silent → 0 items.
      - Healthy: news with 5 headlines + StockTwits with 12 messages +
        Reddit with active threads → ≥ several items.
    """

    def test_single_news_item_and_silent_socials_counts_as_one(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            count_sentiment_evidence,
        )

        news_block = (
            "## News Headlines\n"
            "- 2026-05-10: Tata Steel Q4 net profit jumps 35% to ₹2,400 Cr\n"
        )
        stocktwits_block = "<no StockTwits messages found for $TATASTEEL>"
        reddit_block = (
            "r/wallstreetbets: <no posts found mentioning TATASTEEL in the past 7 days>\n\n"
            "r/stocks: <no posts found mentioning TATASTEEL in the past 7 days>\n\n"
            "r/investing: <no posts found mentioning TATASTEEL in the past 7 days>"
        )
        count = count_sentiment_evidence(news_block, stocktwits_block, reddit_block)
        assert count == 1

    def test_all_three_sources_silent_counts_as_zero(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            count_sentiment_evidence,
        )

        news_block = "No news headlines available for the period."
        stocktwits_block = "<stocktwits unavailable: ConnectionError>"
        reddit_block = (
            "r/wallstreetbets: <no posts found mentioning XYZ in the past 7 days>\n\n"
            "r/stocks: <no posts found mentioning XYZ in the past 7 days>\n\n"
            "r/investing: <no posts found mentioning XYZ in the past 7 days>"
        )
        assert count_sentiment_evidence(news_block, stocktwits_block, reddit_block) == 0

    def test_multiple_news_and_social_signals_counts_high(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            count_sentiment_evidence,
        )

        news_block = (
            "## News Headlines\n"
            "- 2026-05-10: Headline A\n"
            "- 2026-05-09: Headline B\n"
            "- 2026-05-08: Headline C\n"
            "- 2026-05-07: Headline D\n"
            "- 2026-05-06: Headline E\n"
        )
        stocktwits_block = (
            "12 Bullish / 4 Bearish / 2 Neutral over the past 24 hours.\n\n"
            "- @userA: Loving the breakout here\n"
            "- @userB: Earnings will surprise\n"
            "- @userC: Watch the 200-DMA\n"
        )
        reddit_block = (
            "r/wallstreetbets: 1 post: 'XYZ to the moon' (456 upvotes, 92 comments)\n\n"
            "r/stocks: 1 post: 'XYZ deep dive' (203 upvotes, 41 comments)"
        )
        count = count_sentiment_evidence(news_block, stocktwits_block, reddit_block)
        assert count >= 5  # well above the sparse floor


@pytest.mark.unit
class TestSentimentConfidenceBanner:
    """When the counter returns ≤ 1 the analyst prepends a deterministic
    banner to the prose ``sentiment_report`` so downstream readers see the
    sparse-data caveat without having to parse the body."""

    def test_banner_prepended_when_evidence_count_is_one(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            apply_sentiment_confidence_banner,
        )

        prose = "Overall sentiment direction: Bullish. The Q4 print was strong..."
        out = apply_sentiment_confidence_banner(prose, evidence_count=1)
        assert out.startswith("**Sentiment Confidence**: LOW")
        assert "N=1" in out
        assert prose in out  # original prose preserved verbatim

    def test_banner_prepended_when_evidence_count_is_zero(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            apply_sentiment_confidence_banner,
        )

        prose = "Overall sentiment direction: Neutral. No coverage at all..."
        out = apply_sentiment_confidence_banner(prose, evidence_count=0)
        assert out.startswith("**Sentiment Confidence**: LOW")
        assert "N=0" in out

    def test_no_banner_when_evidence_count_is_two_or_more(self):
        from tradingagents.agents.analysts.sentiment_analyst import (
            apply_sentiment_confidence_banner,
        )

        prose = "Overall sentiment direction: Bullish. 5 headlines and 12 StockTwits..."
        out = apply_sentiment_confidence_banner(prose, evidence_count=5)
        assert "**Sentiment Confidence**: LOW" not in out
        assert out == prose
