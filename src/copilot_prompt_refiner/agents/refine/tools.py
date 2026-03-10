from __future__ import annotations

import json
import re
from typing import Any

from copilot_prompt_refiner.agents.microsoft_agent_framework import AzureOpenAIRequestError


def is_content_filter_error(exc: Exception) -> bool:
    """Detect Azure/OpenAI content-filter style failures."""
    if isinstance(exc, AzureOpenAIRequestError):
        payload = exc.response_json.get("error")
        if isinstance(payload, dict):
            code = str(payload.get("code", "")).lower()
            if "content_filter" in code:
                return True
            inner = payload.get("innererror")
            if isinstance(inner, dict):
                inner_code = str(inner.get("code", "")).lower()
                if "responsibleai" in inner_code:
                    return True

    text = str(exc).lower()
    markers = (
        "content_filter",
        "responsibleaipolicyviolation",
        "jailbreak",
        "filtered",
    )
    return any(marker in text for marker in markers)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from raw text when strict parsing is not guaranteed."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        maybe = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return maybe if isinstance(maybe, dict) else None


def extract_diagnostics_from_payload(
    payload: dict[str, Any],
    route: str | None = None,
    status_code: int | None = None,
) -> str:
    """Format structured Azure error payload into concise diagnostic text."""
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        parts = []
        if status_code is not None:
            parts.append(f"status={status_code}")
        if route:
            parts.append(f"route={route}")
        return " ".join(parts) if parts else "No structured filter details available."

    code = str(error_obj.get("code", "")).strip() or "unknown"
    inner = error_obj.get("innererror")
    inner_code = ""
    categories: list[str] = []
    if isinstance(inner, dict):
        inner_code = str(inner.get("code", "")).strip()
        filter_result = inner.get("content_filter_result")
        if isinstance(filter_result, dict):
            for name, details in filter_result.items():
                if not isinstance(details, dict):
                    continue
                detected = bool(details.get("detected"))
                filtered = bool(details.get("filtered"))
                if detected or filtered:
                    categories.append(
                        f"{name}(detected={str(detected).lower()},filtered={str(filtered).lower()})"
                    )

    parts = []
    if status_code is not None:
        parts.append(f"status={status_code}")
    if route:
        parts.append(f"route={route}")
    parts.append(f"code={code}")
    if inner_code:
        parts.append(f"inner={inner_code}")
    if categories:
        parts.append("categories=" + ",".join(categories))
    return "Content filter details: " + " ".join(parts)


def extract_content_filter_diagnostics(exc: Exception) -> str:
    """Extract human-readable content-filter diagnostics from exceptions."""
    if isinstance(exc, AzureOpenAIRequestError):
        return extract_diagnostics_from_payload(
            exc.response_json,
            route=exc.route_url,
            status_code=exc.status_code,
        )

    maybe_json = extract_json_object(str(exc))
    if maybe_json is not None:
        return extract_diagnostics_from_payload(maybe_json)
    return "No structured filter details available."


def sanitize_for_policy(text: str, max_len: int = 260) -> str:
    """Redact high-risk jailbreak phrases and enforce compact text length."""
    compact = " ".join(text.split())
    filtered = re.sub(
        r"(?i)\b(jailbreak|bypass|ignore\s+previous|override\s+policy|developer\s+mode)\b",
        "[redacted]",
        compact,
    )
    if len(filtered) <= max_len:
        return filtered
    return filtered[: max_len - 3].rstrip() + "..."
