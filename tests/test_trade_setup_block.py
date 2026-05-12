"""Regression tests for the schema-bind-failure banner in
``_build_trade_setup_block`` (cli/main.py).

When the upstream salvage parser in
``tradingagents.agents.utils.structured.invoke_structured_or_freetext`` can't
bind a Trader / Portfolio-Manager response into its schema, it prepends the
``[SCHEMA_BIND_FAILED]`` sentinel onto the raw response text. The CLI
renderer must surface that failure visibly (a banner above the at-a-glance
table) instead of quietly substituting a 50-DMA "anchor" — see beads issue
TradingAgents-fk5 for the production incident motivating this.

These tests pin:
  - the banner is emitted whenever either side carries the sentinel;
  - the obsolete `50-DMA fallback (...)` italic substitution is *gone*;
  - per-row `_(unbound)_` markers appear when sentinel-side fields can't
    be extracted;
  - the happy path (neither sentinel) keeps its prior contract intact.
"""

from __future__ import annotations

import pytest

from cli.main import _build_trade_setup_block
from tradingagents.agents.utils.structured import SCHEMA_BIND_FAILED_SENTINEL


# Reusable fixtures: a well-formed Trader and Portfolio-Manager rendered
# block. These mirror what render_trader_proposal / render_pm_decision emit
# downstream — the renderer parses them via regex against the bolded labels.

GOOD_TRADER_PLAN = (
    "## Trader Proposal\n"
    "**Action**: Buy\n"
    "**Entry Price**: 38.72\n"
    "**Initial Stop**: 35.00\n"
    "**Trailing Stop**: 36.50\n"
    "**Position Sizing**: 2% of portfolio\n"
)

GOOD_PM_DECISION = (
    "## Portfolio Decision\n"
    "**Rating**: Strong Buy\n"
    "**Price Target**: 45.00\n"
    "**Time Horizon**: 3 months\n"
)

KEY_LEVELS_WITH_CLOSE = "Latest close: 38.50\n50-DMA: 36.10\n"


@pytest.mark.unit
class TestSchemaBindFailureBanner:
    def test_trader_sentinel_emits_banner_and_drops_50dma_fallback(self):
        """When trader_plan starts with the sentinel, the renderer must:
        (a) emit a visible banner *above* the table mentioning the Trader,
        (b) NEVER emit the obsolete `50-DMA fallback` substitution string,
        (c) still render the PM-side fields (the PM bound fine).
        """
        sentinel_trader = (
            f"{SCHEMA_BIND_FAILED_SENTINEL}\n"
            "I think we should consider buying near the 50-DMA but I'm not "
            "going to commit to specific levels.\n"
        )
        out = _build_trade_setup_block(
            sentinel_trader,
            GOOD_PM_DECISION,
            KEY_LEVELS_WITH_CLOSE,
        )
        assert out is not None, (
            "block dropped entirely when only the Trader side was unbound; "
            "PM-side fields should still drive a renderable table"
        )
        # (a) Banner present and names the Trader.
        assert "Schema-bind failure" in out, (
            "missing visible banner for the schema-bind failure; the whole "
            "point is the user can no longer miss it"
        )
        assert "Trader" in out.split("Schema-bind failure", 1)[1].split("\n", 1)[0], (
            "banner doesn't identify the Trader as the unbound side"
        )
        # Banner must sit ABOVE the table, not inline.
        banner_idx = out.index("Schema-bind failure")
        table_idx = out.index("Trade Setup at a Glance")
        assert banner_idx < table_idx, (
            "banner must precede the table header so the warning is visible "
            "before the numbers are read"
        )
        # (b) Obsolete fallback is GONE permanently.
        assert "50-DMA fallback" not in out, (
            "the silent 50-DMA fallback substitution must not return — the "
            "banner replaces it as the user-visible signal"
        )
        # (c) PM-side still renders.
        assert "Strong Buy" in out
        assert "45.00" in out

    def test_pm_sentinel_emits_banner_naming_portfolio_manager(self):
        """When only the Portfolio Manager side is unbound, the banner must
        name the Portfolio Manager. Trader-side fields must still render.

        We use "Portfolio Manager" not "PM" — the rest of cli/main.py spells
        it out (see headers like "V. Portfolio Manager Decision"), and the
        banner should match that prose so a casual reader recognises the
        agent immediately.
        """
        sentinel_pm = (
            f"{SCHEMA_BIND_FAILED_SENTINEL}\n"
            "I lean bullish but won't commit to a specific rating.\n"
        )
        out = _build_trade_setup_block(
            GOOD_TRADER_PLAN,
            sentinel_pm,
            KEY_LEVELS_WITH_CLOSE,
        )
        assert out is not None
        assert "Schema-bind failure" in out, "missing banner"
        # Should name the Portfolio Manager, not the Trader, not both.
        banner_line = out.split("Schema-bind failure", 1)[1].split("\n", 1)[0]
        assert "Portfolio Manager" in banner_line, (
            f"banner does not name the Portfolio Manager: {banner_line!r}"
        )
        assert "Trader" not in banner_line, (
            f"banner falsely implicates the Trader: {banner_line!r}"
        )
        # Trader-side fields must still render normally.
        assert "Buy" in out
        assert "38.72" in out

    def test_both_sentinels_emits_single_combined_banner(self):
        """When BOTH sides are unbound, exactly one banner is emitted and it
        names both agents. Two stacked banners would be visual noise and
        suggest two independent failures."""
        sentinel_trader = f"{SCHEMA_BIND_FAILED_SENTINEL}\nFree-text trader.\n"
        sentinel_pm = f"{SCHEMA_BIND_FAILED_SENTINEL}\nFree-text PM.\n"
        out = _build_trade_setup_block(
            sentinel_trader,
            sentinel_pm,
            KEY_LEVELS_WITH_CLOSE,
        )
        assert out is not None
        # Exactly one banner.
        assert out.count("Schema-bind failure") == 1, (
            f"expected exactly one banner; got {out.count('Schema-bind failure')}"
        )
        banner_line = out.split("Schema-bind failure", 1)[1].split("\n", 1)[0]
        assert "Trader and Portfolio Manager" in banner_line, (
            f"combined banner does not name both sides: {banner_line!r}"
        )

    def test_happy_path_emits_no_banner_and_no_50dma_fallback(self):
        """When neither side carries the sentinel, the renderer behaves
        exactly as before EXCEPT the 50-DMA fallback substitution is GONE
        permanently. A Trader who omitted Entry Price simply gets an
        Entry-Price-less table — no silent fake anchor.
        """
        # Trader who omitted Entry Price entirely. key_levels still has a
        # 50-DMA, which previously triggered the silent italic substitution.
        trader_no_entry = (
            "## Trader Proposal\n"
            "**Action**: Hold\n"
            "**Initial Stop**: 35.00\n"
            "**Position Sizing**: 0%\n"
        )
        out = _build_trade_setup_block(
            trader_no_entry,
            GOOD_PM_DECISION,
            KEY_LEVELS_WITH_CLOSE,
        )
        assert out is not None
        assert "Schema-bind failure" not in out, (
            "banner emitted on a happy-path render (no sentinel anywhere)"
        )
        assert "50-DMA fallback" not in out, (
            "the silent 50-DMA fallback substitution must not return on the "
            "happy path either — its deletion is permanent regardless of "
            "whether the sentinel fired"
        )
        # Existing prior contract: skipped rows (Entry Price here) drop out
        # of the table entirely on a non-sentinel path.
        assert "**Entry Price**" not in out, (
            "happy-path skipped fields should drop, not become `_(unbound)_`"
        )
        # And the rest of the report still renders.
        assert "Hold" in out
        assert "Strong Buy" in out

    def test_sentinel_trader_without_extractable_entry_marks_row_unbound(self):
        """The whole point of (4) in the bead: when the sentinel fires AND the
        per-row extraction comes up empty, render `_(unbound)_` so the reader
        can see *which fields* the agent failed to deliver — not silently
        drop the row (the user would never know it was missing) and not
        invent a 50-DMA anchor (the bug we're killing).
        """
        # Trader response is JUST the sentinel + free text with no
        # extractable Entry Price / Action / Stops / Position Sizing.
        sentinel_trader = (
            f"{SCHEMA_BIND_FAILED_SENTINEL}\n"
            "I think this stock looks interesting but I want to wait.\n"
        )
        out = _build_trade_setup_block(
            sentinel_trader,
            GOOD_PM_DECISION,
            KEY_LEVELS_WITH_CLOSE,
        )
        assert out is not None
        # Banner present.
        assert "Schema-bind failure" in out
        # Entry Price row exists AND carries the `_(unbound)_` marker.
        # We check the exact row so a stray "_(unbound)_" elsewhere doesn't
        # silently satisfy the assertion.
        assert "- **Entry Price**: _(unbound)_" in out, (
            f"Entry Price row missing or not marked unbound:\n{out}"
        )
        # And we still didn't reintroduce the 50-DMA fallback.
        assert "50-DMA fallback" not in out
