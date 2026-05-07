"""Tests for the outcome-resolution policy module.

The policy controls how the deferred memory loop measures whether a past
decision worked: the holding window (trading days) and the alpha
benchmark (SPY vs Nifty 500 vs other regional indices).
"""

import pytest

from tradingagents.graph.outcome_policy import (
    DEFAULT_HOLDING_DAYS,
    DEFAULT_HORIZON,
    HOLDING_DAYS_BY_HORIZON,
    OutcomePolicy,
    benchmark_for_ticker,
    holding_days_for_horizon,
    resolve_outcome_policy,
)


# ---------------------------------------------------------------------------
# holding_days_for_horizon
# ---------------------------------------------------------------------------

class TestHoldingDaysForHorizon:

    def test_swing_is_about_one_month(self):
        assert holding_days_for_horizon("swing") == 20

    def test_position_is_about_one_quarter(self):
        assert holding_days_for_horizon("position") == 63

    def test_long_term_is_about_one_trading_year(self):
        assert holding_days_for_horizon("long-term") == 252

    def test_case_insensitive(self):
        assert holding_days_for_horizon("SWING") == 20
        assert holding_days_for_horizon("Position") == 63
        assert holding_days_for_horizon("Long-Term") == 252

    def test_unknown_horizon_falls_back_to_default(self):
        assert holding_days_for_horizon("scalp") == DEFAULT_HOLDING_DAYS

    def test_none_falls_back_to_default(self):
        assert holding_days_for_horizon(None) == DEFAULT_HOLDING_DAYS

    def test_empty_string_falls_back_to_default(self):
        assert holding_days_for_horizon("") == DEFAULT_HOLDING_DAYS

    def test_long_term_strictly_longer_than_position(self):
        """Internal monotonicity: longer horizon, longer window."""
        assert (
            HOLDING_DAYS_BY_HORIZON["swing"]
            < HOLDING_DAYS_BY_HORIZON["position"]
            < HOLDING_DAYS_BY_HORIZON["long-term"]
        )


# ---------------------------------------------------------------------------
# benchmark_for_ticker
# ---------------------------------------------------------------------------

class TestBenchmarkForTicker:

    def test_us_default_is_spy(self):
        assert benchmark_for_ticker("NVDA") == "SPY"
        assert benchmark_for_ticker("AAPL") == "SPY"

    def test_nse_uses_nifty_500(self):
        assert benchmark_for_ticker("RELIANCE.NS") == "^CRSLDX"

    def test_bse_uses_nifty_500(self):
        assert benchmark_for_ticker("TCS.BO") == "^CRSLDX"

    def test_lse_uses_ftse(self):
        assert benchmark_for_ticker("AZN.L") == "^FTSE"

    def test_hk_uses_hsi(self):
        assert benchmark_for_ticker("0700.HK") == "^HSI"

    def test_tokyo_uses_nikkei(self):
        assert benchmark_for_ticker("7203.T") == "^N225"

    def test_toronto_uses_tsx_composite(self):
        assert benchmark_for_ticker("RY.TO") == "^GSPTSE"

    def test_asx_uses_all_ords(self):
        assert benchmark_for_ticker("BHP.AX") == "^AXJO"

    def test_case_insensitive(self):
        assert benchmark_for_ticker("reliance.ns") == "^CRSLDX"
        assert benchmark_for_ticker("AzN.l") == "^FTSE"

    def test_none_falls_back_to_spy(self):
        assert benchmark_for_ticker(None) == "SPY"

    def test_empty_falls_back_to_spy(self):
        assert benchmark_for_ticker("") == "SPY"


# ---------------------------------------------------------------------------
# resolve_outcome_policy
# ---------------------------------------------------------------------------

class TestResolveOutcomePolicy:

    def test_swing_us_default(self):
        p = resolve_outcome_policy({"trading_horizon": "swing"}, "NVDA")
        assert p == OutcomePolicy(horizon="swing", holding_days=20, benchmark="SPY")

    def test_position_indian_uses_nifty_500_and_quarter(self):
        p = resolve_outcome_policy({"trading_horizon": "position"}, "RELIANCE.NS")
        assert p.horizon == "position"
        assert p.holding_days == 63
        assert p.benchmark == "^CRSLDX"

    def test_long_term_us_uses_year_window_and_spy(self):
        p = resolve_outcome_policy({"trading_horizon": "long-term"}, "MSFT")
        assert p.horizon == "long-term"
        assert p.holding_days == 252
        assert p.benchmark == "SPY"

    def test_missing_config_uses_defaults(self):
        p = resolve_outcome_policy(None, "NVDA")
        assert p.horizon == DEFAULT_HORIZON
        assert p.holding_days == DEFAULT_HOLDING_DAYS
        assert p.benchmark == "SPY"

    def test_empty_config_uses_defaults(self):
        p = resolve_outcome_policy({}, "NVDA")
        assert p.horizon == DEFAULT_HORIZON
        assert p.holding_days == DEFAULT_HOLDING_DAYS

    def test_unknown_horizon_canonicalised_to_default(self):
        """An invalid horizon value should resolve to the swing default
        rather than leaving the field as a free-form string. This keeps
        downstream prompt rendering predictable."""
        p = resolve_outcome_policy({"trading_horizon": "scalp"}, "NVDA")
        assert p.horizon == DEFAULT_HORIZON
        assert p.holding_days == DEFAULT_HOLDING_DAYS

    def test_horizon_canonicalised_to_lowercase(self):
        p = resolve_outcome_policy({"trading_horizon": "Position"}, "NVDA")
        assert p.horizon == "position"

    def test_outcome_policy_is_frozen(self):
        """OutcomePolicy is a frozen dataclass — assigning to a field raises."""
        p = resolve_outcome_policy({"trading_horizon": "swing"}, "NVDA")
        with pytest.raises(Exception):
            p.holding_days = 999  # type: ignore[misc]
