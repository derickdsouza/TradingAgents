"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the first structured call fails (malformed JSON from a
   weak local model, transient provider issue), we retry up to
   ``max_retries`` times, each time prepending a schema-reminder block
   that quotes the prior validation error and re-asserts the target
   schema. Only after exhausting retries do we fall back to a plain
   ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from tradingagents.agents.utils.freetext_salvage import salvage_into_schema

logger = logging.getLogger(__name__)

# Sentinel string returned when the structured retry loop has exhausted AND
# the salvage parser could not recover a populated schema instance from the
# free-text response. Begins with a fixed token so downstream renderers can
# detect it and surface a loud banner instead of silently substituting
# fallback values. Wiring of the banner is sibling bead fk5 — this module
# only emits the sentinel; it does not detect or render it.
SCHEMA_BIND_FAILED_SENTINEL = "[SCHEMA_BIND_FAILED]"

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _default_max_retries() -> int:
    """Read the retry budget from the env var, falling back to 2."""
    raw = os.environ.get("TRADINGAGENTS_STRUCTURED_RETRIES")
    if raw is None:
        return 2
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _schema_reminder(schema: type[BaseModel], exc: BaseException) -> str:
    """Build a short reminder describing the JSON schema and the prior failure.

    The body is plain text so it works equally for string prompts and for
    chat-message prompts (where we wrap it in a system message).
    """
    try:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
    except Exception:  # pragma: no cover - defensive: any schema introspection issue
        schema_json = schema.__name__
    return (
        f"Your previous response did not bind to the {schema.__name__} schema. "
        f"Reply ONLY with valid JSON conforming to it. "
        f"Validation error: {exc}\n\n"
        f"Schema:\n{schema_json}"
    )


def _prepend_reminder(prompt: Any, reminder: str) -> Any:
    """Return a new prompt with the reminder prepended.

    Handles both supported shapes:
      - ``str``: prepends the reminder as a leading block separated by a blank line.
      - ``list`` of message dicts: prepends a ``system`` message carrying the reminder.
    Any other shape is returned unchanged so we never corrupt an exotic prompt;
    the retry will simply not carry the reminder.
    """
    if isinstance(prompt, str):
        return f"{reminder}\n\n{prompt}"
    if isinstance(prompt, list):
        return [{"role": "system", "content": reminder}, *prompt]
    return prompt


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    schema: Optional[type[BaseModel]] = None,
    max_retries: Optional[int] = None,
) -> str:
    """Run the structured call with bounded retries, then fall back to free-text.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    ``schema`` is the Pydantic model the structured LLM is bound to; it is
    used to inject a schema-reminder block on retry. If omitted, retries are
    still attempted but without the reminder (best-effort).

    ``max_retries`` is the number of *retries* after the initial attempt
    (so total attempts == ``max_retries + 1``). Defaults to the value of the
    ``TRADINGAGENTS_STRUCTURED_RETRIES`` env var, or 2.
    """
    retry_exhausted = False
    if structured_llm is not None:
        retries = max_retries if max_retries is not None else _default_max_retries()
        total_attempts = retries + 1
        last_exc: Optional[BaseException] = None
        current_prompt = prompt
        for attempt in range(1, total_attempts + 1):
            try:
                result = structured_llm.invoke(current_prompt)
                return render(result)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s: structured-output attempt %d/%d failed (%s)",
                    agent_name, attempt, total_attempts, exc,
                )
                if attempt < total_attempts and schema is not None:
                    current_prompt = _prepend_reminder(
                        prompt, _schema_reminder(schema, exc)
                    )
        retry_exhausted = True
        logger.warning(
            "%s: exhausted %d structured attempts (last error: %s); "
            "falling back to free text",
            agent_name, total_attempts, last_exc,
        )

    response = plain_llm.invoke(prompt)
    response_text = response.content

    # Salvage is only attempted when the structured retry loop ran AND
    # exhausted — the ``structured_llm is None`` case (binding never
    # succeeded at all) preserves the legacy raw-content return so
    # provider-agnostic callers that never wanted structured output keep
    # working unchanged.
    if retry_exhausted and schema is not None:
        salvaged = salvage_into_schema(response_text, schema)
        if salvaged is not None:
            logger.info(
                "%s: salvaged free-text response into %s",
                agent_name, schema.__name__,
            )
            return render(salvaged)
        logger.warning(
            "%s: salvage parser could not bind free text to %s; "
            "emitting %s sentinel",
            agent_name, schema.__name__, SCHEMA_BIND_FAILED_SENTINEL,
        )
        return f"{SCHEMA_BIND_FAILED_SENTINEL}\n{response_text}"

    return response_text
