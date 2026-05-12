"""Tests for ``invoke_structured_or_freetext``'s retry-with-schema-reminder loop.

Bead TradingAgents-9md: a one-shot structured call collapses to free text on
the first malformed response. We add a bounded retry loop with a schema
reminder injected on subsequent attempts. The free-text fallback path is
preserved as a final safety net (sibling bead 9av will tighten its salvage).
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from tradingagents.agents.utils.structured import invoke_structured_or_freetext


class _DummySchema(BaseModel):
    """Tiny schema used to verify schema-reminder injection on retry."""

    action: str = Field(..., description="One of BUY/HOLD/SELL")
    reasoning: str = Field(..., description="One-sentence justification")


def _render(obj: _DummySchema) -> str:
    return f"action={obj.action} reasoning={obj.reasoning}"


@pytest.mark.unit
class TestRetryWithSchemaReminder:
    def test_succeeds_on_second_attempt_after_one_failure(self):
        """First call raises; second call must succeed AND its prompt must include
        a schema-reminder line referencing the validation error from attempt 1.
        """
        captured_prompts: list = []
        good = _DummySchema(action="BUY", reasoning="Trend intact.")

        def _invoke(prompt):
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                raise ValueError("missing required field 'reasoning'")
            return good

        structured = MagicMock()
        structured.invoke.side_effect = _invoke

        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content="SHOULD NOT BE USED")

        result = invoke_structured_or_freetext(
            structured,
            plain,
            "Decide on NVDA.",
            _render,
            "TestAgent",
            schema=_DummySchema,
        )

        assert result == "action=BUY reasoning=Trend intact."
        assert len(captured_prompts) == 2, "second attempt must have fired"
        # The free-text fallback must NOT have been touched.
        plain.invoke.assert_not_called()
        # Second prompt must carry a schema reminder that quotes the failure.
        second = captured_prompts[1]
        assert isinstance(second, str)
        assert "missing required field" in second
        # And reference the schema by name so the model knows what to bind to.
        assert "_DummySchema" in second or "DummySchema" in second

    def test_succeeds_on_third_attempt_with_max_retries_two(self):
        """max_retries=2 means total 3 attempts. Fail twice, succeed on the third."""
        attempts: list = []
        good = _DummySchema(action="SELL", reasoning="Guidance cut.")

        def _invoke(prompt):
            attempts.append(prompt)
            if len(attempts) < 3:
                raise ValueError(f"bad-json-attempt-{len(attempts)}")
            return good

        structured = MagicMock()
        structured.invoke.side_effect = _invoke
        plain = MagicMock()

        result = invoke_structured_or_freetext(
            structured,
            plain,
            "Decide on NVDA.",
            _render,
            "TestAgent",
            schema=_DummySchema,
            max_retries=2,
        )

        assert result == "action=SELL reasoning=Guidance cut."
        assert len(attempts) == 3
        plain.invoke.assert_not_called()
        # Each retry's prompt should carry a reminder mentioning the prior error.
        assert "bad-json-attempt-1" in attempts[1]
        assert "bad-json-attempt-2" in attempts[2]

    def test_falls_back_to_freetext_after_exhausting_retries(self):
        """All 3 attempts (1 + 2 retries) fail → free-text path is used."""
        attempts: list = []

        def _invoke(prompt):
            attempts.append(prompt)
            raise ValueError("still bad json")

        structured = MagicMock()
        structured.invoke.side_effect = _invoke

        plain_content = (
            "**Action**: Buy\nReasoning: best we could do as free text."
        )
        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content=plain_content)

        original_prompt = "Decide on NVDA."
        result = invoke_structured_or_freetext(
            structured,
            plain,
            original_prompt,
            _render,
            "TestAgent",
            schema=_DummySchema,
            max_retries=2,
        )

        assert result == plain_content
        assert len(attempts) == 3, "must try initial + 2 retries before falling back"
        # The free-text fallback must see the ORIGINAL prompt (unwrapped),
        # not one of the reminder-prepended retry prompts.
        plain.invoke.assert_called_once_with(original_prompt)

    def test_env_var_zero_disables_retries(self, monkeypatch):
        """TRADINGAGENTS_STRUCTURED_RETRIES=0 → one attempt, no retry, immediate fallback.

        Preserves the legacy one-shot behavior for users who want it.
        """
        monkeypatch.setenv("TRADINGAGENTS_STRUCTURED_RETRIES", "0")
        attempts: list = []

        def _invoke(prompt):
            attempts.append(prompt)
            raise ValueError("malformed")

        structured = MagicMock()
        structured.invoke.side_effect = _invoke

        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content="fallback content")

        result = invoke_structured_or_freetext(
            structured,
            plain,
            "Decide on NVDA.",
            _render,
            "TestAgent",
            schema=_DummySchema,
        )

        assert result == "fallback content"
        assert len(attempts) == 1, "env=0 must disable all retries"
        plain.invoke.assert_called_once()
