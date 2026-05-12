"""Salvage parser for free-text fallback responses.

Bead TradingAgents-9av. When ``invoke_structured_or_freetext``'s retry loop
exhausts, the caller previously returned the plain LLM's free text verbatim.
Downstream renderers greped ``key: value`` lines out of that prose with silent
fallbacks that masked structural failures (e.g. missing ``Entry Price`` would
quietly substitute the 50-DMA). This module replaces the silent path with an
explicit salvage attempt; the caller emits a loud sentinel on failure.

Three paths, tried in order: (1) ``json.loads`` on the whole response, (2)
the first ```json ... ``` fenced block, (3) regex ``key: value`` extraction
keyed off the schema's field names (the case that matters for weak local
models like qwen3.6 that emit key:value blobs rather than JSON). Each path
validates via ``schema.model_validate``; a successful parse must populate at
least ``MIN_SALVAGED_FIELDS`` fields, otherwise a couple of Optional fields
defaulting cleanly could yield a wrong-but-valid instance.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional, TypeVar, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Minimum schema fields that must be populated in the salvaged dict for the
# parse to count. Below this we return None and let the caller emit the
# ``[SCHEMA_BIND_FAILED]`` sentinel rather than risk a false-positive
# parse where only a couple of Optional fields happened to default cleanly.
MIN_SALVAGED_FIELDS = 3


def salvage_into_schema(
    response_text: str, schema: type[T]
) -> Optional[T]:
    """Attempt to parse free-text response into a populated ``schema`` instance.

    Returns the validated instance on success, ``None`` on any failure (parse
    error, validation error, or fewer than ``MIN_SALVAGED_FIELDS`` fields
    successfully extracted). The caller is responsible for emitting the
    ``[SCHEMA_BIND_FAILED]`` sentinel on ``None``.
    """
    if not response_text or not response_text.strip():
        return None

    # Path 1: raw JSON.
    candidate = _try_raw_json(response_text, schema)
    if candidate is not None:
        return candidate

    # Path 2: ```json ... ``` fenced block embedded in prose.
    candidate = _try_fenced_json(response_text, schema)
    if candidate is not None:
        return candidate

    # Path 3: key:value extraction keyed off the schema's field names.
    candidate = _try_key_value_extraction(response_text, schema)
    if candidate is not None:
        return candidate

    return None


def _try_raw_json(text: str, schema: type[T]) -> Optional[T]:
    """Try ``json.loads`` on the whole response, then validate."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return _validate_with_guardrail(data, schema)


_FENCED_JSON_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _try_fenced_json(text: str, schema: type[T]) -> Optional[T]:
    """Try the first ``` ```json ... ``` `` fenced block, then validate."""
    match = _FENCED_JSON_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return _validate_with_guardrail(data, schema)


def _validate_with_guardrail(data: dict, schema: type[T]) -> Optional[T]:
    """Validate ``data`` against ``schema`` and enforce the N>=3 guardrail.

    Counts the number of schema fields explicitly present in ``data`` (matched
    against canonical names and aliases) — NOT the number of Optional fields
    that defaulted to None / a default value. Without this, a dict with one
    or two fields could pass validation through default population.
    """
    if _count_schema_fields(data, schema) < MIN_SALVAGED_FIELDS:
        return None
    try:
        return schema.model_validate(data)
    except Exception:
        return None


def _count_schema_fields(data: dict, schema: type[BaseModel]) -> int:
    """Count keys in ``data`` that match a known field name or alias of ``schema``."""
    known: set[str] = set()
    for field_name, field_info in schema.model_fields.items():
        known.add(field_name)
        alias = getattr(field_info, "alias", None)
        if alias:
            known.add(alias)
    return sum(1 for key in data if key in known)


# Informal synonyms that local models often emit instead of the canonical
# schema field name. These are SALVAGE-only — they do not change the schema's
# JSON contract. Keep this list tight: each entry is justified by a real
# captured failure (e.g. ``rationale`` was emitted by qwen3.6 for the SOUTHBANK
# Run 2 blob where the canonical field is ``reasoning``).
_INFORMAL_SYNONYMS: dict[str, str] = {
    "rationale": "reasoning",
    "sizing": "position_sizing",
    "target": "price_target_horizon",
    "price_target": "price_target_horizon",
    "horizon_target": "price_target_horizon",
}

# Strips ``**bold**`` and inline backticks before key:value matching. The
# bead notes this is sufficient — no full markdown parser needed.
_MARKDOWN_NOISE_RE = re.compile(r"\*+|`+")

# Matches ``key: value`` on a single line. Key allows letters / digits /
# underscore; everything after the first colon is the raw value.
_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")


def _try_key_value_extraction(text: str, schema: type[T]) -> Optional[T]:
    """Extract schema field values from ``key: value`` lines.

    Strips markdown noise, matches against field names + aliases + informal
    synonyms, and coerces values to the field type. Per-field coercion
    failures drop the field (rather than fail the whole salvage); the
    N>=3 guardrail catches the false-positive case.
    """
    field_map = _build_field_map(schema)
    if not field_map:
        return None

    salvaged: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = _MARKDOWN_NOISE_RE.sub("", raw_line)
        match = _KV_RE.match(line)
        if match is None:
            continue
        key_raw, value_raw = match.group(1), match.group(2).strip()
        canonical = field_map.get(key_raw.lower())
        if canonical is None:
            continue
        # Skip if we already captured this field — first occurrence wins,
        # avoids later prose lines overwriting an earlier structured value.
        if canonical in salvaged:
            continue
        field_info = schema.model_fields[canonical]
        coerced = _coerce_value(value_raw, field_info.annotation)
        if coerced is None and not _is_string_field(field_info.annotation):
            # Coercion failed AND the field isn't a free-string field —
            # dropping is safer than letting Pydantic reject the dict whole.
            continue
        salvaged[canonical] = coerced if coerced is not None else value_raw

    return _validate_with_guardrail(salvaged, schema)


def _build_field_map(schema: type[BaseModel]) -> dict[str, str]:
    """Lowercased lookup from field-name / alias / synonym to canonical name.

    Synonyms are mapped only if the canonical target exists on this schema,
    so irrelevant entries aren't smuggled onto schemas that don't need them.
    """
    mapping: dict[str, str] = {}
    for name, info in schema.model_fields.items():
        mapping[name.lower()] = name
        alias = getattr(info, "alias", None)
        if alias:
            mapping[alias.lower()] = name
    for synonym, canonical in _INFORMAL_SYNONYMS.items():
        if canonical in schema.model_fields and synonym.lower() not in mapping:
            mapping[synonym.lower()] = canonical
    return mapping


def _unwrap_optional(annotation):
    """Return the inner type for ``Optional[X]`` (i.e. ``Union[X, None]``)."""
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _is_string_field(annotation) -> bool:
    inner = _unwrap_optional(annotation)
    return inner is str


def _coerce_value(raw: str, annotation) -> Optional[object]:
    """Coerce ``raw`` to match ``annotation``. Returns None on failure.

    Numeric values often arrive with trailing units (``38.72 INR``,
    ``24 months``) or in compound forms (``30%+30%``); strip a leading
    numeric token before parsing for ``float`` / ``int`` fields. Enum
    fields try value match then case-insensitive name match.
    """
    inner = _unwrap_optional(annotation)
    if inner is float:
        return _coerce_float(raw)
    if inner is int:
        return _coerce_int(raw)
    if isinstance(inner, type) and issubclass(inner, Enum):
        return _coerce_enum(raw, inner)
    if inner is str:
        return raw
    # Unknown / complex types (nested models, lists): let model_validate
    # try with the raw string, but fall back to None so caller can drop.
    return None


_LEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_float(raw: str) -> Optional[float]:
    match = _LEADING_NUMBER_RE.search(raw)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _coerce_int(raw: str) -> Optional[int]:
    match = _LEADING_NUMBER_RE.search(raw)
    if match is None:
        return None
    try:
        return int(float(match.group(0)))
    except ValueError:
        return None


def _coerce_enum(raw: str, enum_cls: type[Enum]) -> Optional[Enum]:
    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    # Try exact value match first (case-insensitive on the value).
    for member in enum_cls:
        if str(member.value).lower() == raw_lower:
            return member
    # Then by enum name (e.g. "BUY" -> TraderAction.BUY whose value is "Buy").
    for member in enum_cls:
        if member.name.lower() == raw_lower:
            return member
    return None
