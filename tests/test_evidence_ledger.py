"""Tests for the structured Evidence Ledger.

The ledger is a typed surface of decision-critical facts threaded through
agent state alongside the prose reports. These tests pin the construction,
rendering, and partial-data behavior; the per-agent prompt-inclusion tests
live in tests/test_structured_agents.py once the consumer wiring lands.
"""

import pytest

from tradingagents.agents.utils.evidence_ledger import (
    EvidenceFact,
    EvidenceLedger,
    build_initial_ledger,
    merge_facts,
    render_evidence_ledger,
)


@pytest.mark.unit
class TestEvidenceFact:
    def test_minimal_fact_only_requires_source(self):
        fact = EvidenceFact(value=200.50, source="yfinance")
        assert fact.value == 200.50
        assert fact.source == "yfinance"
        assert fact.as_of is None

    def test_string_value_allowed(self):
        fact = EvidenceFact(value="BULLISH-FAN", source="market_analyst")
        assert fact.value == "BULLISH-FAN"

    def test_optional_as_of_carries_iso_date(self):
        fact = EvidenceFact(value=200.0, source="yfinance", as_of="2026-05-07")
        assert fact.as_of == "2026-05-07"


@pytest.mark.unit
class TestEvidenceLedgerConstruction:
    def test_empty_ledger_only_requires_ticker_and_date(self):
        ledger = EvidenceLedger(ticker="NVDA", trade_date="2026-05-07")
        assert ledger.ticker == "NVDA"
        assert ledger.trade_date == "2026-05-07"
        assert ledger.latest_close is None
        assert ledger.sma_50 is None
        assert ledger.regime_summary is None
        assert ledger.extras == {}

    def test_ledger_carries_typed_facts(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.0, source="yfinance"),
            sma_50=EvidenceFact(value=190.0, source="yfinance"),
        )
        assert ledger.latest_close.value == 200.0
        assert ledger.sma_50.source == "yfinance"

    def test_extras_accept_arbitrary_facts(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            extras={"chandelier_long": EvidenceFact(value=185.0, source="market_analyst")},
        )
        assert ledger.extras["chandelier_long"].value == 185.0


@pytest.mark.unit
class TestRenderEvidenceLedger:
    def test_empty_ledger_renders_empty_string(self):
        ledger = EvidenceLedger(ticker="NVDA", trade_date="2026-05-07")
        assert render_evidence_ledger(ledger) == ""

    def test_full_ledger_renders_all_price_anchors(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.50, source="yfinance"),
            sma_50=EvidenceFact(value=195.40, source="yfinance"),
            sma_200=EvidenceFact(value=180.20, source="yfinance"),
            high_52w=EvidenceFact(value=220.50, source="yfinance"),
            low_52w=EvidenceFact(value=150.00, source="yfinance"),
            high_20d=EvidenceFact(value=205.00, source="yfinance"),
            low_20d=EvidenceFact(value=188.00, source="yfinance"),
        )
        md = render_evidence_ledger(ledger)
        assert "**Evidence Ledger**" in md
        assert "Latest close: 200.50" in md
        assert "50-DMA: 195.40" in md
        assert "200-DMA: 180.20" in md
        assert "52-week range: 150.00 – 220.50" in md
        assert "20-day range: 188.00 – 205.00" in md

    def test_partial_ledger_omits_missing_fields(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            latest_close=EvidenceFact(value=200.0, source="yfinance"),
        )
        md = render_evidence_ledger(ledger)
        assert "Latest close: 200.00" in md
        assert "50-DMA" not in md
        assert "52-week range" not in md

    def test_regime_summary_renders_when_present(self):
        ledger = EvidenceLedger(
            ticker="NIFTY.NS",
            trade_date="2026-05-07",
            regime_summary="**Regime**: Nifty 50 BULLISH | RS-line: outperforming",
        )
        md = render_evidence_ledger(ledger)
        assert "Nifty 50 BULLISH" in md

    def test_extras_render_with_source_attribution(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            extras={
                "chandelier_long": EvidenceFact(value=185.0, source="market_analyst"),
            },
        )
        md = render_evidence_ledger(ledger)
        assert "chandelier_long: 185.0" in md
        assert "(market_analyst)" in md


@pytest.mark.unit
class TestMergeFacts:
    def test_merge_adds_new_extras_without_clobbering_existing(self):
        ledger = EvidenceLedger(
            ticker="NVDA",
            trade_date="2026-05-07",
            extras={"a": EvidenceFact(value=1.0, source="x")},
        )
        merged = merge_facts(ledger, {"b": EvidenceFact(value=2.0, source="y")})
        assert merged.extras["a"].value == 1.0
        assert merged.extras["b"].value == 2.0

    def test_merge_returns_a_new_ledger_without_mutating_input(self):
        ledger = EvidenceLedger(ticker="NVDA", trade_date="2026-05-07")
        merge_facts(ledger, {"x": EvidenceFact(value=1.0, source="s")})
        assert ledger.extras == {}


@pytest.mark.unit
class TestBuildInitialLedger:
    def test_returns_ledger_with_ticker_and_date_even_when_fetch_fails(self, monkeypatch):
        # When the yfinance fetch returns None (malformed date / network error),
        # the ledger must still construct with ticker+trade_date and empty
        # facts — graceful degradation is part of the seam contract.
        monkeypatch.setattr(
            "tradingagents.agents.utils.market_levels._fetch_levels_data",
            lambda ticker, trade_date: None,
        )
        ledger = build_initial_ledger("NVDA", "2026-05-07")
        assert ledger.ticker == "NVDA"
        assert ledger.trade_date == "2026-05-07"
        assert ledger.latest_close is None

    def test_populates_price_anchors_from_fetched_data(self, monkeypatch):
        fake = {
            "latest_close": 200.50,
            "sma_50": 195.40,
            "sma_200": 180.20,
            "high_52w": 220.50,
            "low_52w": 150.00,
            "high_20d": 205.00,
            "low_20d": 188.00,
        }
        monkeypatch.setattr(
            "tradingagents.agents.utils.market_levels._fetch_levels_data",
            lambda ticker, trade_date: fake,
        )
        ledger = build_initial_ledger("NVDA", "2026-05-07")
        assert ledger.latest_close.value == 200.50
        assert ledger.sma_50.value == 195.40
        assert ledger.sma_200.value == 180.20
        assert ledger.high_52w.value == 220.50
        assert ledger.low_52w.value == 150.00
        assert ledger.high_20d.value == 205.00
        assert ledger.low_20d.value == 188.00
        # Source is yfinance for everything in the initial fetch
        assert ledger.latest_close.source == "yfinance"

    def test_partial_data_leaves_missing_fields_none(self, monkeypatch):
        # Short-history tickers may miss sma_200 etc. — those must come back
        # as None on the ledger rather than throwing.
        fake = {
            "latest_close": 200.50,
            "sma_50": 195.40,
            "sma_200": None,
            "high_52w": 220.50,
            "low_52w": 150.00,
            "high_20d": 205.00,
            "low_20d": 188.00,
        }
        monkeypatch.setattr(
            "tradingagents.agents.utils.market_levels._fetch_levels_data",
            lambda ticker, trade_date: fake,
        )
        ledger = build_initial_ledger("NVDA", "2026-05-07")
        assert ledger.latest_close.value == 200.50
        assert ledger.sma_200 is None
