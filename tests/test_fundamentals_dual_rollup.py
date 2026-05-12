"""Tests for the dual-rollup contract on the fundamentals analyst prompt (qyn).

Run 1 of the analyst (Tata Steel, May 2026) rendered quarterly tables only:
Q4 FY26 NII ₹915 Cr, OCF/NI ratio table, provision-to-NI 29.3%→21.4%.
Run 2 of the same analyst rendered annual FY22–FY26 tables only:
FY26 net income ₹14,556 Cr. Same yfinance source, different aggregation.

Annual alone hides the recent inflection (provisioning cool-off across the
trailing four quarters); quarterly alone hides the multi-year trajectory
shape. Either single rollup is incomplete for a trading decision; together
they are load-bearing.

This is a prompt-shape test: code, not the LLM, ensures the analyst is
asked for BOTH a quarterly *and* an annual table. The schema work happens
upstream (the tools already expose ``freq=quarterly|annual``); the gap is
that the analyst was previously free to render either.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestFundamentalsDualRollupPromptContract:
    """The fundamentals system message must:
      1. Explicitly require BOTH a quarterly AND an annual rollup table.
      2. Specify trailing horizons (4 quarters / 5 FYs) so the model
         doesn't pick an arbitrary window.
      3. Carry a fallback clause that names the missing-rollup language
         to use when source data is single-rollup only.
    """

    def test_system_message_demands_quarterly_table(self):
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="TATASTEEL.NS", lookback_phrase="the past 90 days"
        )
        assert "quarterly" in msg.lower()
        # the requirement is about a *rendered table*, not just freely
        # mentioning the word — anchor on table-shape language
        assert "trailing 4 quarter" in msg.lower() or "last 4 quarter" in msg.lower()

    def test_system_message_demands_annual_table(self):
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="TATASTEEL.NS", lookback_phrase="the past 90 days"
        )
        assert "annual" in msg.lower()
        assert "trailing 5 fy" in msg.lower() or "last 5 fy" in msg.lower() \
            or "trailing 5 fiscal year" in msg.lower()

    def test_system_message_mentions_both_rollups_explicitly(self):
        """The two-rollup requirement should not be implicit — a future
        prompt edit that quietly drops one rollup should make this fail."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="AAPL", lookback_phrase="the past 90 days"
        ).lower()
        # both words present AND co-located in the spirit of "both X and Y"
        assert "both" in msg
        assert "quarterly" in msg and "annual" in msg

    def test_system_message_specifies_qoq_and_yoy_columns(self):
        """The quarterly table is only useful with comparison columns —
        a bare quarterly snapshot doesn't reveal the inflection."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="AAPL", lookback_phrase="the past 90 days"
        ).lower()
        assert "qoq" in msg
        assert "yoy" in msg

    def test_system_message_carries_missing_rollup_fallback_clause(self):
        """When the source returns single-rollup data the analyst must
        say so explicitly, not silently pick one. Anchor on the phrase
        that downstream readers grep for."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="AAPL", lookback_phrase="the past 90 days"
        )
        assert "not available" in msg.lower()
        # the deterministic phrase the analyst is told to emit
        assert "only annual rollup shown" in msg.lower() \
            or "only quarterly rollup shown" in msg.lower()

    def test_indian_ticker_still_gets_dual_rollup_requirement(self):
        """The Indian-ticker shareholding-pattern clause is additive; it
        must not displace the dual-rollup requirement."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="TATASTEEL.NS", lookback_phrase="the past 90 days"
        ).lower()
        assert "quarterly" in msg and "annual" in msg
        # the Indian-specific shareholding clause still present
        assert "shareholding" in msg or "promoter" in msg

    def test_non_indian_ticker_does_not_get_shareholding_clause(self):
        from tradingagents.agents.analysts.fundamentals_analyst import (
            build_fundamentals_system_message,
        )

        msg = build_fundamentals_system_message(
            ticker="AAPL", lookback_phrase="the past 90 days"
        ).lower()
        assert "promoter pledge" not in msg
