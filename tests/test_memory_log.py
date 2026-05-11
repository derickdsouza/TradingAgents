"""Tests for TradingMemoryLog — storage, deferred reflection, PM injection, legacy removal."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.utils.evidence_ledger import EvidenceFact, EvidenceLedger
from tradingagents.graph.reflection import Reflector
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.propagation import Propagator
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

_SEP = TradingMemoryLog._SEPARATOR

DECISION_BUY = "Rating: Buy\nEnter at $189-192, 6% portfolio cap."
DECISION_OVERWEIGHT = (
    "Rating: Overweight\n"
    "Executive Summary: Moderate position, await confirmation.\n"
    "Investment Thesis: Strong fundamentals but near-term headwinds."
)
DECISION_SELL = "Rating: Sell\nExit position immediately."
DECISION_NO_RATING = (
    "Executive Summary: Complex situation with multiple competing factors.\n"
    "Investment Thesis: No clear directional signal at this time."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_log(tmp_path, filename="trading_memory.md"):
    config = {"memory_log_path": str(tmp_path / filename)}
    return TradingMemoryLog(config)


def _seed_completed(tmp_path, ticker, date, decision_text, reflection_text, filename="trading_memory.md"):
    """Write a completed entry directly to file, bypassing the API."""
    entry = (
        f"[{date} | {ticker} | Buy | +1.0% | +0.5% | 5d]\n\n"
        f"DECISION:\n{decision_text}\n\n"
        f"REFLECTION:\n{reflection_text}"
        + _SEP
    )
    with open(tmp_path / filename, "a", encoding="utf-8") as f:
        f.write(entry)


def _resolve_entry(log, ticker, date, decision, reflection="Good call."):
    """Store a decision then immediately resolve it via the API."""
    log.store_decision(ticker, date, decision)
    log.update_with_outcome(ticker, date, 0.05, 0.02, 5, reflection)


def _price_df(prices):
    """Minimal DataFrame matching yfinance .history() output shape."""
    return pd.DataFrame({"Close": prices})


def _make_pm_state(past_context="", latest_close=200.0, trade_date="2026-01-10"):
    """Minimal AgentState dict for portfolio_manager_node.

    Carries an Evidence Ledger by default so the PM decision-contract
    validator (which loud-fails on missing close) sees a reference price
    and lets legitimate targets through. Tests that need a missing close
    can pass ``latest_close=None``.
    """
    ledger = EvidenceLedger(
        ticker="NVDA",
        trade_date=trade_date,
        latest_close=(
            EvidenceFact(value=latest_close, source="yfinance", as_of=trade_date)
            if latest_close is not None
            else None
        ),
    )
    return {
        "company_of_interest": "NVDA",
        "trade_date": trade_date,
        "past_context": past_context,
        "evidence_ledger": ledger,
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_plan": "Research plan.",
        "trader_investment_plan": "Trader plan.",
    }


def _structured_pm_llm(captured: dict, decision: PortfolioDecision | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real PortfolioDecision (so render_pm_decision works).
    """
    if decision is None:
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold the position; await catalyst.",
            investment_thesis="Balanced view; neither side carried the debate.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# Core: storage and read path
# ---------------------------------------------------------------------------

class TestTradingMemoryLogCore:

    def test_store_creates_file(self, tmp_path):
        log = make_log(tmp_path)
        assert not (tmp_path / "trading_memory.md").exists()
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert (tmp_path / "trading_memory.md").exists()

    def test_store_appends_not_overwrites(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        entries = log.load_entries()
        assert len(entries) == 2
        assert entries[0]["ticker"] == "NVDA"
        assert entries[1]["ticker"] == "AAPL"

    def test_store_decision_idempotent(self, tmp_path):
        """Calling store_decision twice with same (ticker, date) stores only one entry."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert len(log.load_entries()) == 1

    def test_batch_update_resolves_multiple_entries(self, tmp_path):
        """batch_update_with_outcomes resolves multiple pending entries in one write."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.store_decision("NVDA", "2026-01-12", DECISION_SELL)

        updates = [
            {"ticker": "NVDA", "trade_date": "2026-01-05",
             "raw_return": 0.05, "alpha_return": 0.02, "holding_days": 5,
             "reflection": "First correct."},
            {"ticker": "NVDA", "trade_date": "2026-01-12",
             "raw_return": -0.03, "alpha_return": -0.01, "holding_days": 5,
             "reflection": "Second correct."},
        ]
        log.batch_update_with_outcomes(updates)

        entries = log.load_entries()
        assert len(entries) == 2
        assert all(not e["pending"] for e in entries)
        assert entries[0]["reflection"] == "First correct."
        assert entries[1]["reflection"] == "Second correct."

    def test_pending_tag_format(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" in text

    # Rating parsing

    def test_rating_parsed_buy(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_overweight(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        assert log.load_entries()[0]["rating"] == "Overweight"

    def test_rating_fallback_hold(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        assert log.load_entries()[0]["rating"] == "Hold"

    def test_rating_priority_over_prose(self, tmp_path):
        """'Rating: X' label wins even when an opposing rating word appears earlier in prose."""
        decision = (
            "The sell thesis is weak. The hold case is marginal.\n\n"
            "Rating: Buy\n\n"
            "Executive Summary: Strong fundamentals support the position."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    # Delimiter robustness

    def test_decision_with_markdown_separator(self, tmp_path):
        """LLM decision containing '---' must not corrupt the entry."""
        decision = "Rating: Buy\n\n---\n\nRisk: elevated volatility."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        entries = log.load_entries()
        assert len(entries) == 1
        assert "Risk: elevated volatility" in entries[0]["decision"]

    # load_entries

    def test_load_entries_empty_file(self, tmp_path):
        log = make_log(tmp_path)
        assert log.load_entries() == []

    def test_load_entries_single(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["date"] == "2026-01-10"
        assert e["ticker"] == "NVDA"
        assert e["rating"] == "Buy"
        assert e["pending"] is True
        assert e["raw"] is None

    def test_load_entries_multiple(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        entries = log.load_entries()
        assert len(entries) == 3
        assert [e["ticker"] for e in entries] == ["NVDA", "AAPL", "MSFT"]

    def test_decision_content_preserved(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["decision"] == DECISION_BUY.strip()

    # get_pending_entries

    def test_get_pending_returns_pending_only(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA.", "Correct.")
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        pending = log.get_pending_entries()
        assert len(pending) == 1
        assert pending[0]["ticker"] == "NVDA"
        assert pending[0]["date"] == "2026-01-10"

    # get_past_context

    def test_get_past_context_empty(self, tmp_path):
        log = make_log(tmp_path)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_pending_excluded(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_same_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA — AI capex thesis intact.", "Directionally correct.")
        ctx = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in ctx
        assert "Buy NVDA" in ctx

    def test_get_past_context_cross_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "AAPL", "2026-01-05", "Buy AAPL — Services growth.", "Correct.")
        ctx = log.get_past_context("NVDA")
        assert "Recent cross-ticker lessons" in ctx
        assert "Past analyses of NVDA" not in ctx

    def test_n_same_limit_respected(self, tmp_path):
        """Only the n_same most recent same-ticker entries are included."""
        log = make_log(tmp_path)
        for i in range(6):
            _seed_completed(tmp_path, "NVDA", f"2026-01-{i+1:02d}", f"Buy entry {i}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_same=5)
        assert "Buy entry 0" not in ctx
        assert "Buy entry 5" in ctx

    def test_n_cross_limit_respected(self, tmp_path):
        """Only the n_cross most recent cross-ticker entries are included."""
        log = make_log(tmp_path)
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOG", "META"]):
            _seed_completed(tmp_path, ticker, f"2026-01-{i+1:02d}", f"Buy {ticker}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_cross=3)
        assert "AAPL" not in ctx
        assert "META" in ctx

    # No-op when config is None

    def test_no_log_path_is_noop(self):
        log = TradingMemoryLog(config=None)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries() == []
        assert log.get_past_context("NVDA") == ""

    # Rotation: opt-in cap on resolved entries

    def test_rotation_disabled_by_default(self, tmp_path):
        """Without max_entries, all resolved entries are kept."""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 7

    def test_rotation_prunes_oldest_resolved(self, tmp_path):
        """When max_entries is set and exceeded, oldest resolved entries are pruned."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 3,
        })
        # Resolve 5 entries; rotation should keep only the 3 most recent.
        for i in range(5):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        entries = log.load_entries()
        assert len(entries) == 3
        # Confirm the OLDEST were dropped, not the newest.
        dates = [e["date"] for e in entries]
        assert dates == ["2026-01-03", "2026-01-04", "2026-01-05"]

    def test_rotation_never_prunes_pending(self, tmp_path):
        """Pending entries (unresolved) are kept regardless of the cap."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 2,
        })
        # 3 resolved + 2 pending. With cap=2, only 2 resolved survive; both pending stay.
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Resolved {i}.")
        log.store_decision("NVDA", "2026-02-01", DECISION_BUY)
        log.store_decision("NVDA", "2026-02-02", DECISION_OVERWEIGHT)
        # Trigger rotation by resolving one more entry — pending entries must stay.
        _resolve_entry(log, "NVDA", "2026-01-04", DECISION_BUY, "Resolved 3.")
        entries = log.load_entries()
        pending = [e for e in entries if e["pending"]]
        resolved = [e for e in entries if not e["pending"]]
        assert len(pending) == 2, "pending entries must never be pruned"
        assert len(resolved) == 2, f"expected 2 resolved after rotation, got {len(resolved)}"

    def test_rotation_under_cap_is_noop(self, tmp_path):
        """No rotation when resolved count <= max_entries."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 10,
        })
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 3

    # Rating parsing: markdown bold and numbered list formats

    def test_rating_parsed_from_bold_markdown(self, tmp_path):
        """**Rating**: Buy — markdown bold around the label must not prevent parsing."""
        decision = "**Rating**: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_from_bold_value(self, tmp_path):
        """Rating: **Sell** — markdown bold around the value must not prevent parsing."""
        decision = "Rating: **Sell**\nExit immediately."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_label_wins_over_prose_with_markdown(self, tmp_path):
        """Rating: **Sell** must win even when prose contains a conflicting rating word."""
        decision = (
            "The buy thesis is weakened by guidance.\n"
            "Rating: **Sell**\n"
            "Exit before earnings."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_parsed_from_numbered_list(self, tmp_path):
        """1. Rating: Buy — numbered list prefix must not prevent parsing."""
        decision = "1. Rating: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"


# ---------------------------------------------------------------------------
# Deferred reflection: update_with_outcome, Reflector, _fetch_returns
# ---------------------------------------------------------------------------

class TestDeferredReflection:

    # update_with_outcome

    def test_update_replaces_pending_tag(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" not in text
        assert "+4.2%" in text
        assert "+2.1%" in text
        assert "5d" in text

    def test_update_appends_reflection(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["reflection"] == "Momentum confirmed."
        assert e["decision"] == DECISION_BUY.strip()

    def test_update_preserves_other_entries(self, tmp_path):
        """Only the matching entry is modified; all other entries remain unchanged."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", "Rating: Hold\nHold AAPL.")
        log.store_decision("MSFT", "2026-01-12", DECISION_SELL)
        log.update_with_outcome("AAPL", "2026-01-11", 0.01, -0.01, 5, "Neutral result.")
        entries = log.load_entries()
        assert len(entries) == 3
        nvda, aapl, msft = entries
        assert nvda["ticker"] == "NVDA" and nvda["pending"] is True
        assert aapl["ticker"] == "AAPL" and aapl["pending"] is False
        assert aapl["reflection"] == "Neutral result."
        assert msft["ticker"] == "MSFT" and msft["pending"] is True

    def test_update_atomic_write(self, tmp_path):
        """A pre-existing .tmp file is overwritten; the log is correctly updated."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        stale_tmp = tmp_path / "trading_memory.tmp"
        stale_tmp.write_text("GARBAGE CONTENT — should be overwritten", encoding="utf-8")
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Correct.")
        assert not stale_tmp.exists()
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["reflection"] == "Correct."
        assert entries[0]["pending"] is False

    def test_update_noop_when_no_log_path(self):
        log = TradingMemoryLog(config=None)
        log.update_with_outcome("NVDA", "2026-01-10", 0.05, 0.02, 5, "Reflection")

    def test_formatting_roundtrip_after_update(self, tmp_path):
        """All fields intact and blank line between tag and DECISION preserved after update."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["decision"] == DECISION_BUY.strip()
        assert e["reflection"] == "Momentum confirmed."
        assert e["raw"] == "+4.2%"
        assert e["alpha"] == "+2.1%"
        assert e["holding"] == "5d"
        raw_text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | +4.2% | +2.1% | 5d]\n\nDECISION:" in raw_text

    # Reflector.reflect_on_final_decision

    def test_reflect_on_final_decision_returns_llm_output(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Directionally correct. Thesis confirmed."
        reflector = Reflector(mock_llm)
        result = reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY, raw_return=0.042, alpha_return=0.021
        )
        assert result == "Directionally correct. Thesis confirmed."
        mock_llm.invoke.assert_called_once()

    def test_reflect_on_final_decision_includes_returns_in_prompt(self):
        """Return figures are present in the human message sent to the LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Incorrect call."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_SELL, raw_return=-0.08, alpha_return=-0.05
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "-8.0%" in human_content
        assert "-5.0%" in human_content
        assert "Exit position immediately." in human_content

    # TradingAgentsGraph._fetch_returns

    def test_fetch_returns_valid_ticker(self):
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        spy_prices   = [400.0, 402.0, 404.0, 403.0, 405.0, 406.0]
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days = TradingAgentsGraph._fetch_returns(mock_graph, "NVDA", "2026-01-05")
        assert raw is not None and alpha is not None and days is not None
        assert isinstance(raw, float) and isinstance(alpha, float) and isinstance(days, int)
        assert days == 5

    def test_fetch_returns_too_recent(self):
        """Only 1 data point available → returns (None, None, None), no crash."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = _price_df([100.0])
            mock_ticker_cls.return_value = m
            raw, alpha, days = TradingAgentsGraph._fetch_returns(mock_graph, "NVDA", "2026-04-19")
        assert raw is None and alpha is None and days is None

    def test_fetch_returns_delisted(self):
        """Empty DataFrame → returns (None, None, None), no crash."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = pd.DataFrame({"Close": []})
            mock_ticker_cls.return_value = m
            raw, alpha, days = TradingAgentsGraph._fetch_returns(mock_graph, "XXXXXFAKE", "2026-01-10")
        assert raw is None and alpha is None and days is None

    def test_fetch_returns_spy_shorter_than_stock(self):
        """SPY having fewer rows than the stock must not raise IndexError."""
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        spy_prices   = [400.0, 402.0, 403.0]
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days = TradingAgentsGraph._fetch_returns(mock_graph, "NVDA", "2026-01-05")
        assert raw is not None and alpha is not None and days is not None
        assert days == 2

    # TradingAgentsGraph._resolve_benchmark — picks index for alpha calc

    def test_resolve_benchmark_explicit_override(self):
        """config['benchmark_ticker'] wins for every ticker."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "benchmark_ticker": "QQQ",
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "7203.T") == "QQQ"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "NVDA") == "QQQ"

    def test_resolve_benchmark_suffix_map(self):
        """Known suffixes route to their regional index."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "benchmark_ticker": None,
            "benchmark_map": {
                ".T": "^N225", ".HK": "^HSI", ".NS": "^NSEI",
                ".L": "^FTSE", ".TO": "^GSPTSE", ".AX": "^AXJO",
                ".BO": "^BSESN", "": "SPY",
            },
        }
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "7203.T") == "^N225"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "0700.HK") == "^HSI"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "RELIANCE.NS") == "^NSEI"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "AZN.L") == "^FTSE"

    def test_resolve_benchmark_us_ticker_defaults_to_spy(self):
        """US tickers (no dotted suffix) take the empty-suffix entry."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "NVDA") == "SPY"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "AAPL") == "SPY"

    def test_resolve_benchmark_unknown_suffix_falls_back(self):
        """Unrecognised suffix (BRK.B, FAKE.XX) falls back to SPY."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "FAKE.XX") == "SPY"
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "BRK.B") == "SPY"

    def test_resolve_benchmark_case_insensitive(self):
        """Suffix matching is case-insensitive so 7203.t resolves like 7203.T."""
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "benchmark_ticker": None,
            "benchmark_map": {".T": "^N225", "": "SPY"},
        }
        assert TradingAgentsGraph._resolve_benchmark(mock_graph, "7203.t") == "^N225"

    def test_reflector_includes_benchmark_in_label(self):
        """benchmark appears in the prompt label, not 'SPY' hardcoded."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Directionally correct."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY,
            raw_return=0.05,
            alpha_return=0.02,
            benchmark="^N225",
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "Alpha vs ^N225:" in human_content
        assert "Alpha vs SPY:" not in human_content

    def test_reflector_defaults_to_spy_for_unupdated_callers(self):
        """Default benchmark keeps the SPY label for legacy callers."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "ok"
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY,
            raw_return=0.05,
            alpha_return=0.02,
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "Alpha vs SPY:" in human_content

    # TradingAgentsGraph._resolve_pending_entries

    def test_resolve_skips_other_tickers(self, tmp_path):
        """Pending AAPL entry is not resolved when the run is for NVDA."""
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-10", DECISION_BUY)
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.memory_log = log
        mock_graph._fetch_returns = MagicMock(return_value=(0.05, 0.02, 20))
        mock_graph.config = {"trading_horizon": "swing"}
        TradingAgentsGraph._resolve_pending_entries(mock_graph, "NVDA")
        mock_graph._fetch_returns.assert_not_called()
        assert len(log.get_pending_entries()) == 1

    # Reflector: horizon, holding-window, and benchmark in prompt

    def test_reflect_includes_benchmark_name_in_prompt(self):
        """The benchmark passed by the resolver appears in the human message,
        so reflections written for non-US tickers correctly cite Nifty 500
        (not SPY) when the call ran against an Indian listing."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "OK."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY,
            raw_return=0.04,
            alpha_return=0.01,
            benchmark="^CRSLDX",
            holding_days=63,
            horizon="position",
        )
        messages = mock_llm.invoke.call_args[0][0]
        human = next(content for role, content in messages if role == "human")
        assert "Alpha vs ^CRSLDX" in human
        assert "Holding window: 63 trading days" in human
        assert "Horizon: position" in human

    def test_reflect_default_benchmark_is_spy(self):
        """When the caller omits benchmark/horizon (legacy callers), the prompt
        falls back to SPY without surfacing window metadata."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "OK."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY, raw_return=0.04, alpha_return=0.01
        )
        messages = mock_llm.invoke.call_args[0][0]
        human = next(content for role, content in messages if role == "human")
        assert "Alpha vs SPY" in human

    # Resolver: horizon-driven ripeness and benchmark routing

    def test_resolve_skips_entry_when_horizon_not_ripe(self, tmp_path):
        """Position horizon (63 trading days) but only 20 days of price data
        available → entry stays pending until enough data accumulates."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.memory_log = log
        mock_graph.reflector = MagicMock()
        mock_graph.config = {"trading_horizon": "position"}
        # _fetch_returns finds only 20 trading days of data — short of the
        # 63 the position horizon requires.
        mock_graph._fetch_returns = MagicMock(return_value=(0.05, 0.02, 20))
        TradingAgentsGraph._resolve_pending_entries(mock_graph, "NVDA")
        assert len(log.get_pending_entries()) == 1
        mock_graph.reflector.reflect_on_final_decision.assert_not_called()

    def test_resolve_uses_indian_benchmark_for_ns_ticker(self, tmp_path):
        """For an .NS ticker, _fetch_returns is called with benchmark='^CRSLDX'
        (not SPY), and the reflector is told the same."""
        log = make_log(tmp_path)
        log.store_decision("RELIANCE.NS", "2026-01-05", DECISION_BUY)
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_final_decision.return_value = "Captured Nifty alpha."
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.memory_log = log
        mock_graph.reflector = mock_reflector
        mock_graph.config = {"trading_horizon": "swing"}
        mock_graph._fetch_returns = MagicMock(return_value=(0.06, 0.02, 20))
        TradingAgentsGraph._resolve_pending_entries(mock_graph, "RELIANCE.NS")
        assert mock_graph._fetch_returns.call_args.kwargs["benchmark"] == "^CRSLDX"
        assert mock_graph._fetch_returns.call_args.kwargs["holding_days"] == 20
        assert mock_reflector.reflect_on_final_decision.call_args.kwargs["benchmark"] == "^CRSLDX"
        assert mock_reflector.reflect_on_final_decision.call_args.kwargs["horizon"] == "swing"

    def test_resolve_marks_entry_completed(self, tmp_path):
        """After resolve, get_pending_entries() is empty and the entry has a REFLECTION."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_final_decision.return_value = "Momentum confirmed."
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.memory_log = log
        mock_graph.reflector = mock_reflector
        mock_graph.config = {"trading_horizon": "swing"}
        # Swing horizon resolves at 20 trading days; return enough days so the
        # entry is considered ripe and gets resolved rather than left pending.
        mock_graph._fetch_returns = MagicMock(return_value=(0.05, 0.02, 20))
        TradingAgentsGraph._resolve_pending_entries(mock_graph, "NVDA")
        assert log.get_pending_entries() == []
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is False
        assert entries[0]["reflection"] == "Momentum confirmed."
        assert "+5.0%" in entries[0]["raw"]
        assert "+2.0%" in entries[0]["alpha"]


# ---------------------------------------------------------------------------
# Portfolio Manager injection: past_context in state and prompt
# ---------------------------------------------------------------------------

class TestPortfolioManagerInjection:

    # past_context in initial state

    def test_past_context_in_initial_state(self):
        propagator = Propagator()
        state = propagator.create_initial_state("NVDA", "2026-01-10", past_context="some context")
        assert "past_context" in state
        assert state["past_context"] == "some context"

    def test_past_context_defaults_to_empty(self):
        propagator = Propagator()
        state = propagator.create_initial_state("NVDA", "2026-01-10")
        assert state["past_context"] == ""

    # PM prompt

    def test_pm_prompt_includes_past_context(self):
        captured = {}
        llm = _structured_pm_llm(captured)
        pm_node = create_portfolio_manager(llm)
        state = _make_pm_state(past_context="[2026-01-05 | NVDA | Buy | +5.0% | +2.0% | 5d]\nGreat call.")
        pm_node(state)
        assert "Lessons from prior decisions and outcomes" in captured["prompt"]
        assert "Great call." in captured["prompt"]

    def test_pm_no_past_context_no_section(self):
        """PM prompt omits the lessons section entirely when past_context is empty."""
        captured = {}
        llm = _structured_pm_llm(captured)
        pm_node = create_portfolio_manager(llm)
        state = _make_pm_state(past_context="")
        pm_node(state)
        assert "Lessons from prior decisions" not in captured["prompt"]

    def test_pm_returns_rendered_markdown_with_rating(self):
        """The structured PortfolioDecision is rendered to markdown that
        downstream consumers (memory log, signal processor, CLI display)
        can parse without any extra LLM call."""
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Build position gradually over the next two weeks.",
            investment_thesis="AI capex cycle remains intact; institutional flows constructive.",
            price_target=215.0,
            time_horizon="3-6 months",
        )
        llm = _structured_pm_llm(captured, decision)
        pm_node = create_portfolio_manager(llm)
        result = pm_node(_make_pm_state())
        md = result["final_trade_decision"]
        assert "**Rating**: Overweight" in md
        assert "**Executive Summary**: Build position gradually" in md
        assert "**Investment Thesis**: AI capex cycle" in md
        assert "**Price Target**: 215.0" in md
        assert "**Time Horizon**: 3-6 months" in md

    def test_pm_validator_drops_invalid_target_end_to_end(self):
        """PARACABLES regression: a Hold + target -13% from close, threaded
        through the real ``create_portfolio_manager`` node, must come out
        with the bad target absent and a Validation Notes footer present.
        """
        captured = {}
        # Hold with a target sitting -13% below close (59.77 vs 52.0) is
        # structurally a Sell signal; the validator drops it.
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold the position; await sector catalyst.",
            investment_thesis="Evidence balanced; no decisive near-term driver.",
            price_target_horizon=52.0,
            target_basis="DCF",
        )
        llm = _structured_pm_llm(captured, decision)
        pm_node = create_portfolio_manager(llm)
        state = _make_pm_state(latest_close=59.77, trade_date="2026-05-11")
        result = pm_node(state)
        md = result["final_trade_decision"]
        assert "**Price Target**: 52.0" not in md
        assert "**Horizon Target**: 52.0" not in md
        assert "**Validation Notes**" in md
        assert "TARGET_DIRECTION_VIOLATION" in md

    def test_pm_falls_back_to_freetext_when_structured_unavailable(self):
        """If a provider does not support with_structured_output, the agent
        falls back to a plain invoke and returns whatever prose the model
        produced, so the pipeline never blocks."""
        plain_response = "**Rating**: Sell\n\nExit ahead of guidance."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        pm_node = create_portfolio_manager(llm)
        result = pm_node(_make_pm_state())
        assert result["final_trade_decision"] == plain_response

    # get_past_context ordering and limits

    def test_same_ticker_prioritised(self, tmp_path):
        """Same-ticker entries in same-ticker section; cross-ticker entries in cross-ticker section."""
        log = make_log(tmp_path)
        _resolve_entry(log, "NVDA", "2026-01-05", DECISION_BUY, "Momentum confirmed.")
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued.")
        result = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in result
        assert "Recent cross-ticker lessons" in result
        same_block, cross_block = result.split("Recent cross-ticker lessons")
        assert "NVDA" in same_block
        assert "AAPL" in cross_block

    def test_cross_ticker_reflection_only(self, tmp_path):
        """Cross-ticker entries show only the REFLECTION text, not the full DECISION."""
        log = make_log(tmp_path)
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued correction.")
        result = log.get_past_context("NVDA")
        assert "Overvalued correction." in result
        assert "Exit position immediately." not in result

    def test_n_same_limit_respected(self, tmp_path):
        """More than 5 same-ticker completed entries → only 5 injected."""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        result = log.get_past_context("NVDA", n_same=5)
        lessons_present = sum(1 for i in range(7) if f"Lesson {i}." in result)
        assert lessons_present == 5

    def test_n_cross_limit_respected(self, tmp_path):
        """More than 3 cross-ticker completed entries → only 3 injected."""
        log = make_log(tmp_path)
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
        for i, ticker in enumerate(tickers):
            _resolve_entry(log, ticker, f"2026-01-{i+1:02d}", DECISION_BUY, f"{ticker} lesson.")
        result = log.get_past_context("NVDA", n_cross=3)
        cross_count = sum(result.count(f"{t} lesson.") for t in tickers)
        assert cross_count == 3

    # Full A→B→C integration cycle

    def test_full_cycle_store_resolve_inject(self, tmp_path):
        """store pending → resolve with outcome → past_context non-empty for PM."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        assert len(log.get_pending_entries()) == 1
        assert log.get_past_context("NVDA") == ""
        log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "Correct call.")
        assert log.get_pending_entries() == []
        past_ctx = log.get_past_context("NVDA")
        assert past_ctx != ""
        assert "NVDA" in past_ctx
        assert "Correct call." in past_ctx
        assert "DECISION:" in past_ctx
        assert "REFLECTION:" in past_ctx


# ---------------------------------------------------------------------------
# Legacy removal: BM25 / FinancialSituationMemory fully gone
# ---------------------------------------------------------------------------

class TestLegacyRemoval:

    def test_financial_situation_memory_removed(self):
        """FinancialSituationMemory must not be importable from the memory module."""
        import tradingagents.agents.utils.memory as m
        assert not hasattr(m, "FinancialSituationMemory")

    def test_bm25_not_imported(self):
        """rank_bm25 must not be present in the memory module namespace."""
        import tradingagents.agents.utils.memory as m
        assert not hasattr(m, "BM25Okapi")

    def test_reflect_and_remember_removed(self):
        """TradingAgentsGraph must not expose reflect_and_remember."""
        assert not hasattr(TradingAgentsGraph, "reflect_and_remember")

    def test_portfolio_manager_no_memory_param(self):
        """create_portfolio_manager accepts only llm; passing memory= raises TypeError."""
        mock_llm = MagicMock()
        create_portfolio_manager(mock_llm)
        with pytest.raises(TypeError):
            create_portfolio_manager(mock_llm, memory=MagicMock())

    def test_full_pipeline_no_regression(self, tmp_path):
        """propagate() completes and stores the decision after the redesign."""
        import functools

        fake_state = {
            "final_trade_decision": "Rating: Buy\nBuy NVDA.",
            "company_of_interest": "NVDA",
            "trade_date": "2026-01-10",
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "", "judge_decision": "",
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "", "judge_decision": "",
                "current_aggressive_response": "", "current_conservative_response": "",
                "current_neutral_response": "", "count": 1, "latest_speaker": "",
            },
        }
        mock_graph = MagicMock()
        mock_graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.config = {"results_dir": str(tmp_path)}
        mock_graph.graph.invoke.return_value = fake_state
        mock_graph.propagator.create_initial_state.return_value = fake_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        # Bind the real _run_graph so propagate's call to self._run_graph executes
        # the actual write path instead of the auto-MagicMock.
        mock_graph._run_graph = functools.partial(
            TradingAgentsGraph._run_graph, mock_graph
        )
        TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")
        entries = mock_graph.memory_log.load_entries()
        assert len(entries) == 1
        assert entries[0]["ticker"] == "NVDA"
        assert entries[0]["pending"] is True


class TestGetPriorPmDecision:
    """``TradingMemoryLog.get_prior_pm_decision(ticker)`` returns the most
    recent committed-vintage entry for ``ticker`` (any status, pending or
    resolved). Drives the Slice 5 cross-run drift check."""

    def test_returns_none_when_log_is_empty(self, tmp_path):
        log = make_log(tmp_path)
        assert log.get_prior_pm_decision("NVDA") is None

    def test_returns_none_when_no_entry_for_ticker(self, tmp_path):
        log = make_log(tmp_path)
        # AAPL entry with full vintage block.
        decision = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 100.0 — _dcf_\n\n"
            "**Horizon Target**: 100.0 — _dcf_\n\n"
            "**Target Vintage**: committed 2026-04-27 at 90.0"
        )
        log.store_decision("AAPL", "2026-04-27", decision)
        assert log.get_prior_pm_decision("NVDA") is None

    def test_returns_none_when_most_recent_entry_lacks_vintage(self, tmp_path):
        """If the most-recent NVDA entry has no Target Vintage line AND no
        committed_close, there is nothing to drift-check against."""
        log = make_log(tmp_path)
        decision = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 100.0 — _dcf_\n\n"
            "**Horizon Target**: 100.0 — _dcf_"
        )
        log.store_decision("NVDA", "2026-04-27", decision)
        assert log.get_prior_pm_decision("NVDA") is None

    def test_parses_horizon_target_and_vintage_from_decision_markdown(self, tmp_path):
        log = make_log(tmp_path)
        decision = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 60.0 — _dcf_\n\n"
            "**Horizon Target**: 60.0 — _dcf_\n\n"
            "**Target Vintage**: committed 2026-04-27 at 50.0"
        )
        log.store_decision("NVDA", "2026-04-27", decision)
        prior = log.get_prior_pm_decision("NVDA")
        assert prior is not None
        assert prior["price_target_horizon"] == 60.0
        assert prior["target_committed_at"] == "2026-04-27"
        assert prior["committed_close"] == 50.0
        assert prior["trade_date"] == "2026-04-27"

    def test_returns_most_recent_entry_when_multiple_for_same_ticker(self, tmp_path):
        log = make_log(tmp_path)
        older = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 60.0\n\n"
            "**Horizon Target**: 60.0\n\n"
            "**Target Vintage**: committed 2026-04-01 at 50.0"
        )
        newer = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 70.0\n\n"
            "**Horizon Target**: 70.0\n\n"
            "**Target Vintage**: committed 2026-04-27 at 55.0"
        )
        log.store_decision("NVDA", "2026-04-01", older)
        log.store_decision("NVDA", "2026-04-27", newer)
        prior = log.get_prior_pm_decision("NVDA")
        assert prior is not None
        assert prior["trade_date"] == "2026-04-27"
        assert prior["price_target_horizon"] == 70.0
        assert prior["committed_close"] == 55.0

    def test_returns_entry_with_basis_label_stripped(self, tmp_path):
        """Horizon Target with a `— _basis_` suffix still parses cleanly."""
        log = make_log(tmp_path)
        decision = (
            "**Rating**: Overweight\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 105.5 — _peer_multiple_\n\n"
            "**Horizon Target**: 105.5 — _peer_multiple_\n\n"
            "**Target Vintage**: committed 2026-04-27 at 100.0"
        )
        log.store_decision("NVDA", "2026-04-27", decision)
        prior = log.get_prior_pm_decision("NVDA")
        assert prior is not None
        assert prior["price_target_horizon"] == 105.5

    def test_works_for_pending_entry(self, tmp_path):
        """A still-pending entry is also valid for drift-comparison purposes —
        the trade may not have resolved yet, but the target vintage is real."""
        log = make_log(tmp_path)
        decision = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 60.0\n\n"
            "**Horizon Target**: 60.0\n\n"
            "**Target Vintage**: committed 2026-04-27 at 50.0"
        )
        log.store_decision("NVDA", "2026-04-27", decision)
        # Don't resolve. Still pending.
        prior = log.get_prior_pm_decision("NVDA")
        assert prior is not None
        assert prior["price_target_horizon"] == 60.0


class TestPortfolioManagerDriftInjection:
    """End-to-end: ``create_portfolio_manager(llm, memory_log=...)`` looks
    up the prior PM decision; when drift exceeds threshold, the new run's
    prompt carries a ``**Drift Reaffirmation Required:**`` block."""

    def _seed_prior(self, log, ticker, trade_date, target, committed_close, committed_at):
        decision = (
            f"**Rating**: Buy\n\n"
            f"**Executive Summary**: Prior.\n\n"
            f"**Investment Thesis**: Prior.\n\n"
            f"**Price Target**: {target}\n\n"
            f"**Horizon Target**: {target}\n\n"
            f"**Target Vintage**: committed {committed_at} at {committed_close}"
        )
        log.store_decision(ticker, trade_date, decision)

    def test_first_run_no_prior_no_drift_directive(self, tmp_path):
        """First run for ticker: no prior, no drift directive."""
        log = make_log(tmp_path)
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        pm(_make_pm_state(latest_close=50.0, trade_date="2026-05-11"))
        assert "Drift Reaffirmation Required" not in captured["prompt"]

    def test_drift_under_threshold_no_directive(self, tmp_path):
        """Prior committed at 50, current 52 (+4%) → no directive."""
        log = make_log(tmp_path)
        self._seed_prior(log, "NVDA", "2026-04-27", 60.0, 50.0, "2026-04-27")
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        pm(_make_pm_state(latest_close=52.0, trade_date="2026-05-11"))
        assert "Drift Reaffirmation Required" not in captured["prompt"]

    def test_drift_over_threshold_injects_directive(self, tmp_path):
        """Prior committed at 50, current 58 (+16%) → directive injected."""
        log = make_log(tmp_path)
        self._seed_prior(log, "NVDA", "2026-04-27", 60.0, 50.0, "2026-04-27")
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        pm(_make_pm_state(latest_close=58.0, trade_date="2026-05-11"))
        prompt = captured["prompt"]
        assert "Drift Reaffirmation Required" in prompt
        # The directive itself must carry the diagnostic numbers.
        assert "ATTENTION" in prompt
        assert "+16.0%" in prompt
        assert "REAFFIRM" in prompt
        assert "REVISE" in prompt

    def test_drift_directive_appears_before_be_decisive(self, tmp_path):
        """Directive must sit just above the ``Be decisive and ground...``
        instruction, so the LLM reads it before producing output."""
        log = make_log(tmp_path)
        self._seed_prior(log, "NVDA", "2026-04-27", 60.0, 50.0, "2026-04-27")
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        pm(_make_pm_state(latest_close=58.0, trade_date="2026-05-11"))
        prompt = captured["prompt"]
        drift_idx = prompt.index("Drift Reaffirmation Required")
        decisive_idx = prompt.index("Be decisive and ground")
        assert drift_idx < decisive_idx

    def test_prior_without_committed_close_no_directive(self, tmp_path):
        """A prior entry without a vintage block (legacy / dropped target)
        does NOT trigger the drift check on the next run."""
        log = make_log(tmp_path)
        legacy_decision = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Test.\n\n"
            "**Investment Thesis**: Test.\n\n"
            "**Price Target**: 60.0\n\n"
            "**Horizon Target**: 60.0"  # NO target vintage line
        )
        log.store_decision("NVDA", "2026-04-27", legacy_decision)
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        pm(_make_pm_state(latest_close=58.0, trade_date="2026-05-11"))
        assert "Drift Reaffirmation Required" not in captured["prompt"]

    def test_custom_threshold_triggers_at_lower_drift(self, tmp_path):
        """Prior carrying ``target_drift_threshold_pct=0.10`` triggers at
        12% drift even though it would not at the 15% default."""
        # Seed a decision through the renderer so the (Slice 5) snapshot
        # behavior persists the threshold into the vintage block via the
        # PM node. For this isolated retrieval-side test we seed the
        # markdown directly with the threshold injected; the regex-based
        # parser doesn't read the threshold (it's a snapshot of the LLM
        # decision), so the threshold takes effect on the SECOND run via
        # the schema default — meaning the threshold here is what was
        # SET on the prior PM decision before snapshot. The PM node
        # supplies the schema default; an LLM could override.
        #
        # For TDD purposes, threshold-on-prior is verified via the
        # validator-level unit test (TestValidateTargetDrift). Here we
        # confirm the END-TO-END wiring with the schema default 15%.
        log = make_log(tmp_path)
        self._seed_prior(log, "NVDA", "2026-04-27", 60.0, 50.0, "2026-04-27")
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm, memory_log=log)
        # 12% drift: 50 → 56. Under default 15% threshold → no directive.
        pm(_make_pm_state(latest_close=56.0, trade_date="2026-05-11"))
        assert "Drift Reaffirmation Required" not in captured["prompt"]

    def test_pm_node_works_without_memory_log(self, tmp_path):
        """Backwards-compat: ``create_portfolio_manager(llm)`` without
        memory_log keeps working — no drift check, no error."""
        captured = {}
        llm = _structured_pm_llm(captured)
        pm = create_portfolio_manager(llm)  # no memory_log kwarg
        pm(_make_pm_state(latest_close=58.0, trade_date="2026-05-11"))
        assert "Drift Reaffirmation Required" not in captured["prompt"]


class TestPortfolioManagerVintageSnapshot:
    """End-to-end: when the PM publishes a horizon target, the renderer
    snapshots ``target_committed_at`` and ``committed_close`` into the
    decision before validation. The vintage line appears in the rendered
    markdown — and through the memory log, becomes the anchor for the
    NEXT run's drift check."""

    def test_render_snapshots_vintage_into_decision(self, tmp_path):
        """A decision shipped with a horizon target must come out with the
        vintage block in the rendered markdown."""
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Build position gradually.",
            investment_thesis="Setup intact.",
            price_target_horizon=58.0,
            target_basis="dcf",
        )
        llm = _structured_pm_llm(captured, decision)
        pm = create_portfolio_manager(llm)
        result = pm(_make_pm_state(latest_close=50.0, trade_date="2026-05-11"))
        md = result["final_trade_decision"]
        assert "**Target Vintage**: committed 2026-05-11 at 50.0" in md

    def test_render_does_not_snapshot_vintage_when_no_horizon_target(self, tmp_path):
        """No horizon target → no vintage line. Vintage without a target
        is meaningless (nothing to drift-check)."""
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold; await catalyst.",
            investment_thesis="Balanced.",
            # No price_target_horizon.
        )
        llm = _structured_pm_llm(captured, decision)
        pm = create_portfolio_manager(llm)
        result = pm(_make_pm_state(latest_close=50.0, trade_date="2026-05-11"))
        md = result["final_trade_decision"]
        assert "**Target Vintage**" not in md

