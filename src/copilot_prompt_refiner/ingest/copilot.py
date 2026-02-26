from __future__ import annotations

import ast
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from copilot_prompt_refiner.models import AgentCase, MessageLog


PROMPT_KEY_HINTS = (
    "system_prompt",
    "system-prompt",
    "systemprompt",
    "system_message",
    "system-message",
    "systemmessage",
    "developer_prompt",
    "instructions",
    "prompt",
)

PROMPT_ASSIGNMENT_PATTERN = re.compile(
    r"(?is)\b(?P<key>system[_-]?prompt|system[_-]?message|systemPrompt|systemMessage|developer[_-]?prompt|instructions?|prompt)\b\s*[:=]\s*(?P<quote>\"\"\"|'''|`|\"|')(?P<text>.*?)(?P=quote)"
)

SYSTEM_ROLE_PATTERN = re.compile(
    r"(?is)\brole\b\s*[:=]\s*[\"']system[\"'][^\\n\\r]{0,300}?\\bcontent\b\s*[:=]\s*(?P<quote>\"\"\"|'''|`|\"|')(?P<text>.*?)(?P=quote)"
)

LOG_DISCOVERY_PATTERNS = (
    "logs/**/*.json",
    "**/*log*.json",
    "**/*chat*.json",
    "**/*conversation*.json",
    "**/*copilot*.json",
)

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}


def _message_from_row(row: dict[str, Any]) -> MessageLog:
    """Normalize one raw message row into `MessageLog`.

    The mapper accepts common field aliases (`text`, `message`) so payloads from
    multiple clients can be ingested without per-client adapters.
    """
    return MessageLog(
        role=str(row.get("role", "assistant")),
        content=str(
            row.get("content")
            or row.get("text")
            or row.get("message")
            or ""
        ),
        timestamp=row.get("timestamp"),
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"role", "content", "text", "message", "timestamp"}
        },
    )


def _logs_from_json_payload(payload: Any) -> list[MessageLog]:
    """Parse logs from JSON-compatible payload shapes.

    Supports both list-of-message objects and envelope objects with `messages`,
    returning an empty list when shape is unsupported.
    """
    if isinstance(payload, list):
        return [_message_from_row(row) for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [_message_from_row(row) for row in messages if isinstance(row, dict)]

    return []


def _logs_from_text(text: str) -> list[MessageLog]:
    """Parse lightweight plain-text logs into role/content message records.

    Lines prefixed with `user:` or `assistant:` are mapped to roles; other lines
    default to assistant for backward compatibility.
    """
    logs: list[MessageLog] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        role = "assistant"
        if clean.lower().startswith("user:"):
            role = "user"
            clean = clean[5:].strip()
        elif clean.lower().startswith("assistant:"):
            role = "assistant"
            clean = clean[10:].strip()
        logs.append(MessageLog(role=role, content=clean))
    return logs


def load_logs_from_payload(logs_payload: Any) -> list[MessageLog]:
    """Load logs from dict/list/string payloads with tolerant parsing behavior.

    The function attempts JSON decoding first and falls back to plain-text line
    parsing, allowing ingestion even when payloads are loosely formatted.
    """
    if logs_payload is None:
        return []

    parsed = _logs_from_json_payload(logs_payload)
    if parsed:
        return parsed

    if isinstance(logs_payload, str):
        text = logs_payload.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return _logs_from_text(text)
        parsed_decoded = _logs_from_json_payload(decoded)
        if parsed_decoded:
            return parsed_decoded
        return _logs_from_text(text)

    return []


def _parse_time_like(value: Any) -> float:
    """Parse numeric or ISO-like timestamps into sortable epoch seconds.

    Invalid values return `-1.0` so source ranking can still proceed without
    raising exceptions on malformed metadata.
    """
    if value is None:
        return -1.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return -1.0
        try:
            return float(text)
        except ValueError:
            pass

        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return -1.0

    return -1.0


def _normalize_log_sources(log_sources: Any) -> list[dict[str, Any]]:
    """Normalize heterogeneous log source inputs into comparable entries.

    Output records include path/content/timestamp/index fields so later logic
    can rank recency and choose the best source deterministically.
    """
    if log_sources is None:
        return []

    items: list[Any]
    if isinstance(log_sources, list):
        items = log_sources
    elif isinstance(log_sources, dict):
        maybe_files = log_sources.get("files")
        if isinstance(maybe_files, list):
            items = maybe_files
        else:
            items = [log_sources]
    else:
        items = [log_sources]

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            content = item.get("content") or item.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            path = item.get("path") or item.get("name") or f"log_source_{index}.json"
            timestamp = _parse_time_like(
                item.get("modified_at") or item.get("mtime") or item.get("timestamp")
            )
            normalized.append(
                {
                    "path": str(path),
                    "content": content,
                    "timestamp": timestamp,
                    "index": index,
                }
            )
            continue

        if isinstance(item, str) and item.strip():
            normalized.append(
                {
                    "path": f"log_source_{index}.txt",
                    "content": item,
                    "timestamp": -1.0,
                    "index": index,
                }
            )

    return normalized


def _select_latest_log_source(log_sources: Any) -> dict[str, Any] | None:
    """Select the highest-priority log source from normalized candidates.

    Ranking prioritizes newer timestamps, JSON-like filenames, and log-hinted
    paths to maximize chance of recovering current conversation context.
    """
    normalized = _normalize_log_sources(log_sources)
    if not normalized:
        return None

    def sort_key(entry: dict[str, Any]) -> tuple[float, int, int, int]:
        """Return ranking tuple for latest-log selection heuristics.

        Higher timestamp wins first, then JSON/log-like path hints and original
        list order are used as deterministic tie-breakers.
        """
        path_lower = str(entry.get("path", "")).lower()
        is_json = 1 if path_lower.endswith(".json") else 0
        has_log_hint = 1 if any(
            token in path_lower for token in ("log", "chat", "conversation", "copilot")
        ) else 0
        return (
            float(entry.get("timestamp", -1.0)),
            is_json,
            has_log_hint,
            int(entry.get("index", 0)),
        )

    normalized.sort(key=sort_key, reverse=True)
    return normalized[0]


def _discover_latest_log_text(workspace: str | Path) -> tuple[str | None, str | None]:
    """Discover and read the most recent local log file under workspace.

    Search skips common build/venv/cache directories and returns both file text
    and absolute path so metadata can record provenance.
    """
    root = Path(workspace)
    if not root.exists() or not root.is_dir():
        return None, None

    root_resolved = root.resolve()
    best_path: Path | None = None
    best_mtime = -1.0

    for pattern in LOG_DISCOVERY_PATTERNS:
        for path in root_resolved.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel_parts = path.resolve().relative_to(root_resolved).parts
            except ValueError:
                continue
            if any(part in SKIP_DIR_NAMES for part in rel_parts):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = path

    if best_path is None:
        return None, None

    return best_path.read_text(encoding="utf-8", errors="ignore"), str(best_path.resolve())


def _extract_string_constant(node: ast.AST) -> str | None:
    """Return stripped string literal value from an AST node when available.

    Used by prompt discovery to safely inspect assignments without executing
    source files.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return None


def extract_system_prompt_from_python_source(source: str) -> str | None:
    """Extract best system-prompt candidate from Python source text.

    The extractor scans assignments and dict literals via AST to avoid execution,
    then ranks candidates by key specificity and content length.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates: list[tuple[int, str]] = []

    def register(name: str, value: str | None) -> None:
        """Register a candidate with heuristic priority by variable/key name.

        Priority favors explicit system-prompt identifiers over generic prompt
        keys, improving candidate quality for multi-file repositories.
        """
        if not value:
            return
        normalized = name.lower()
        if normalized in {
            "system_prompt",
            "agent_system_prompt",
            "default_system_prompt",
            "system_message",
        }:
            priority = 0
        elif "system_prompt" in normalized or "system_message" in normalized:
            priority = 1
        elif "prompt" in normalized:
            priority = 2
        else:
            priority = 3
        candidates.append((priority, value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _extract_string_constant(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    register(target.id, value)

            if isinstance(node.value, ast.Dict):
                for key_node, value_node in zip(node.value.keys, node.value.values):
                    key = _extract_string_constant(key_node) or ""
                    value_from_dict = _extract_string_constant(value_node)
                    if key.lower() in {
                        "system_prompt",
                        "system-message",
                        "system_message",
                    }:
                        register(key, value_from_dict)

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _extract_string_constant(node.value) if node.value is not None else None
            register(node.target.id, value)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -len(item[1])))
    return candidates[0][1]


def _normalize_prompt_sources(
    prompt_sources: Any,
    definition_py_content: str | None,
) -> list[dict[str, str]]:
    """Normalize prompt source payloads into `{path, content}` records.

    This function accepts list/dict/string variants and always prepends inline
    `definition_py_content` when provided.
    """
    normalized: list[dict[str, str]] = []

    if isinstance(definition_py_content, str) and definition_py_content.strip():
        normalized.append(
            {
                "path": "definition.py",
                "content": definition_py_content,
            }
        )

    if prompt_sources is None:
        return normalized

    items: list[Any]
    if isinstance(prompt_sources, list):
        items = prompt_sources
    elif isinstance(prompt_sources, dict):
        maybe_files = prompt_sources.get("files")
        if isinstance(maybe_files, list):
            items = maybe_files
        else:
            items = [prompt_sources]
    else:
        items = [prompt_sources]

    for index, item in enumerate(items):
        if isinstance(item, dict):
            content = item.get("content") or item.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            path = item.get("path") or item.get("name") or f"source_{index}"
            normalized.append({"path": str(path), "content": content})
            continue

        if isinstance(item, str) and item.strip():
            normalized.append({"path": f"source_{index}.txt", "content": item})

    return normalized


def _score_prompt_candidate(key: str, text: str, source_path: str, base_score: int) -> int:
    """Heuristically score one prompt candidate for source selection.

    Scoring combines key/path semantics and text length sanity checks to prefer
    likely system prompts while down-weighting noisy or tiny snippets.
    """
    score = base_score
    key_lower = key.lower()
    path_lower = source_path.lower()
    text_len = len(text.strip())

    if key_lower in {"system_prompt", "systemprompt", "system_message", "systemmessage"}:
        score += 55
    elif "system" in key_lower and ("prompt" in key_lower or "message" in key_lower):
        score += 45
    elif "prompt" in key_lower:
        score += 25
    elif "instruction" in key_lower:
        score += 18

    if path_lower.endswith("definition.py"):
        score += 30
    if "definition" in path_lower:
        score += 18
    if "prompt" in path_lower:
        score += 14
    if "agent" in path_lower:
        score += 8

    if text_len < 20:
        score -= 30
    elif text_len < 50:
        score -= 10
    elif text_len > 12000:
        score -= 20

    return score


def _collect_prompt_candidates_from_json(
    value: Any,
    source_path: str,
    depth: int = 0,
) -> list[tuple[int, str, str]]:
    """Recursively collect prompt-like string fields from JSON structures.

    Depth and item limits keep traversal bounded so malformed or massive inputs
    do not cause runaway parsing cost.
    """
    if depth > 6:
        return []

    candidates: list[tuple[int, str, str]] = []

    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if isinstance(nested, str) and any(hint in key_lower for hint in PROMPT_KEY_HINTS):
                score = _score_prompt_candidate(key_text, nested, source_path, base_score=70)
                candidates.append((score, source_path, nested.strip()))
                continue

            candidates.extend(
                _collect_prompt_candidates_from_json(
                    nested,
                    source_path=source_path,
                    depth=depth + 1,
                )
            )
        return candidates

    if isinstance(value, list):
        for nested in value[:80]:
            candidates.extend(
                _collect_prompt_candidates_from_json(
                    nested,
                    source_path=source_path,
                    depth=depth + 1,
                )
            )

    return candidates


def _collect_prompt_candidates_from_source(
    source_path: str,
    content: str,
) -> list[tuple[int, str, str]]:
    """Collect scored prompt candidates from one source file's content.

    Multiple extraction strategies are combined: AST parsing, JSON traversal,
    regex assignment matching, and role-based message patterns.
    """
    candidates: list[tuple[int, str, str]] = []

    python_extracted = extract_system_prompt_from_python_source(content)
    if python_extracted:
        score = _score_prompt_candidate(
            "system_prompt",
            python_extracted,
            source_path,
            base_score=80,
        )
        candidates.append((score, source_path, python_extracted))

    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None
        if decoded is not None:
            candidates.extend(
                _collect_prompt_candidates_from_json(decoded, source_path=source_path)
            )

    for match in PROMPT_ASSIGNMENT_PATTERN.finditer(content):
        key = match.group("key") or "prompt"
        text = (match.group("text") or "").strip()
        if not text:
            continue
        score = _score_prompt_candidate(key, text, source_path, base_score=65)
        candidates.append((score, source_path, text))

    for match in SYSTEM_ROLE_PATTERN.finditer(content):
        text = (match.group("text") or "").strip()
        if not text:
            continue
        score = _score_prompt_candidate(
            "system_message",
            text,
            source_path,
            base_score=75,
        )
        candidates.append((score, source_path, text))

    if not candidates and ("prompt" in source_path.lower() or "definition" in source_path.lower()):
        text = content.strip()
        if text and len(text) <= 12000:
            score = _score_prompt_candidate("prompt_text", text, source_path, base_score=35)
            candidates.append((score, source_path, text))

    return candidates


def _resolve_system_prompt(
    system_prompt: str | None,
    definition_py_content: str | None,
    prompt_sources: Any = None,
) -> tuple[str | None, str | None, int]:
    """Resolve final system prompt text plus provenance metadata.

    Direct input wins; otherwise all source candidates are scored and the top
    result is returned with source label and candidate count.
    """
    if system_prompt and system_prompt.strip():
        return system_prompt.strip(), "input", 1

    normalized_sources = _normalize_prompt_sources(
        prompt_sources=prompt_sources,
        definition_py_content=definition_py_content,
    )

    candidates: list[tuple[int, str, str]] = []
    for source in normalized_sources:
        source_path = source["path"]
        content = source["content"]
        candidates.extend(
            _collect_prompt_candidates_from_source(
                source_path=source_path,
                content=content,
            )
        )

    if not candidates:
        return None, None, 0

    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    top_score, top_source, top_prompt = candidates[0]
    source_label = f"prompt_sources:{top_source}:score={top_score}"
    return top_prompt.strip(), source_label, len(candidates)


def _extract_ground_truth_from_payload(payload: Any) -> str | None:
    """Extract ground-truth text from flexible payload conventions.

    Supports dict keys, raw strings, and JSON-encoded strings so upstream
    tools can provide expected output in multiple formats.
    """
    keys = [
        "ground_truth",
        "expected",
        "expected_output",
        "golden",
        "answer",
    ]

    if payload is None:
        return None

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, str):
            return decoded.strip() or None
        return _extract_ground_truth_from_payload(decoded)

    return None


def _resolve_ground_truth(
    ground_truth: str | None,
    ground_truth_content: Any,
) -> tuple[str | None, str | None]:
    """Resolve ground truth value and its source label.

    Explicit `ground_truth` takes precedence; otherwise extraction is attempted
    from `ground_truth_content` payload fields.
    """
    if isinstance(ground_truth, str) and ground_truth.strip():
        return ground_truth.strip(), "input"

    extracted = _extract_ground_truth_from_payload(ground_truth_content)
    if extracted:
        return extracted, "payload.ground_truth_content"

    return None, None


def _infer_user_input(
    user_input: str | None,
    logs: list[MessageLog],
    logs_source_label: str = "payload.logs",
) -> tuple[str | None, str | None]:
    """Infer user input from explicit value or most recent user log message.

    Source labels are returned with the value to improve discovery metadata
    and troubleshooting for payload completeness issues.
    """
    if isinstance(user_input, str) and user_input.strip():
        return user_input.strip(), "input"

    for message in reversed(logs):
        if message.role.lower() == "user" and message.content.strip():
            return message.content.strip(), logs_source_label

    return None, None


def _resolve_logs_payload(
    logs: Any,
    log_sources: Any,
    workspace: str | Path,
) -> tuple[list[MessageLog], str | None]:
    """Resolve logs from direct payload, log sources, or workspace discovery.

    The function returns parsed logs plus a source label that records where
    the winning log evidence was selected from.
    """
    parsed_logs = load_logs_from_payload(logs)
    if parsed_logs:
        return parsed_logs, "payload.logs"

    latest_log_source = _select_latest_log_source(log_sources)
    if latest_log_source is not None:
        parsed_from_sources = load_logs_from_payload(latest_log_source.get("content"))
        if parsed_from_sources:
            return (
                parsed_from_sources,
                f"payload.log_sources:{latest_log_source.get('path')}",
            )

    local_log_text, local_log_path = _discover_latest_log_text(workspace)
    if local_log_text:
        parsed_local = load_logs_from_payload(local_log_text)
        if parsed_local:
            return parsed_local, f"workspace:{local_log_path}"

    return [], None


def build_case(
    *,
    system_prompt: str,
    user_input: str,
    ground_truth: str | None = None,
    logs: Any = None,
    workspace: str | Path = ".",
    context_files: list[str] | None = None,
    case_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCase:
    """Build a normalized `AgentCase` from explicit caller-provided fields.

    This helper is primarily used by tests and direct integrations that already
    know prompt/input values and only need log/context normalization.
    """
    parsed_logs = load_logs_from_payload(logs)
    merged_context = list(dict.fromkeys(context_files or []))

    base_metadata = {"workspace": str(Path(workspace))}
    if metadata:
        base_metadata.update(metadata)

    return AgentCase(
        case_id=case_id or f"case-{uuid.uuid4().hex[:8]}",
        system_prompt=system_prompt,
        user_input=user_input,
        ground_truth=ground_truth,
        logs=parsed_logs,
        context_files=merged_context,
        metadata=base_metadata,
    )


def build_case_from_payload(
    *,
    workspace: str | Path = ".",
    system_prompt: str | None = None,
    definition_py_content: str | None = None,
    prompt_sources: Any = None,
    user_input: str | None = None,
    require_user_input: bool = True,
    ground_truth: str | None = None,
    ground_truth_content: Any = None,
    logs: Any = None,
    log_sources: Any = None,
    context_files: list[str] | None = None,
    case_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCase:
    """Build `AgentCase` from MCP-style payload with inference and fallbacks.

    The builder resolves prompt, logs, user input, and ground truth from mixed
    payload shapes while preserving rich discovery metadata.
    """
    resolved_system_prompt, system_prompt_source, system_prompt_candidates = _resolve_system_prompt(
        system_prompt=system_prompt,
        definition_py_content=definition_py_content,
        prompt_sources=prompt_sources,
    )
    if not resolved_system_prompt:
        raise ValueError(
            "Could not infer system_prompt. Provide system_prompt or prompt_sources/definition_py_content with prompt definitions."
        )

    parsed_logs, log_source = _resolve_logs_payload(
        logs=logs,
        log_sources=log_sources,
        workspace=workspace,
    )

    if require_user_input and not (isinstance(user_input, str) and user_input.strip()):
        raise ValueError(
            "user_input is required. Pass the current VS Code Copilot chat user input in payload_input.user_input."
        )

    resolved_user_input, user_input_source = _infer_user_input(
        user_input=user_input,
        logs=parsed_logs,
        logs_source_label=log_source or "payload.logs",
    )
    if not resolved_user_input:
        raise ValueError(
            "Could not infer user_input. Provide user_input or include user-role logs in payload.logs."
        )

    resolved_ground_truth, ground_truth_source = _resolve_ground_truth(
        ground_truth=ground_truth,
        ground_truth_content=ground_truth_content,
    )

    merged_context = list(dict.fromkeys(context_files or []))

    base_metadata: dict[str, Any] = {
        "workspace": str(Path(workspace)),
        "discovery": {
            "mode": "payload",
            "system_prompt_source": system_prompt_source,
            "system_prompt_candidates": system_prompt_candidates,
            "user_input_source": user_input_source,
            "user_input_required": require_user_input,
            "ground_truth_source": ground_truth_source,
            "log_source": log_source,
        },
    }
    if metadata:
        base_metadata.update(metadata)

    return AgentCase(
        case_id=case_id or f"case-{uuid.uuid4().hex[:8]}",
        system_prompt=resolved_system_prompt,
        user_input=resolved_user_input,
        ground_truth=resolved_ground_truth,
        logs=parsed_logs,
        context_files=merged_context,
        metadata=base_metadata,
    )


# Backward-compatible wrapper for callers that still pass explicit fields.
def build_case_from_sources(
    *,
    system_prompt: str | None = None,
    system_prompt_file: str | Path | None = None,
    user_input: str,
    ground_truth: str | None = None,
    ground_truth_file: str | Path | None = None,
    log_path: str | Path | None = None,
    workspace: str | Path = ".",
) -> AgentCase:
    """Backward-compatible wrapper for legacy file-path based call sites.

    It reads optional prompt/ground-truth/log files and delegates to `build_case`
    so older integrations keep working with the newer payload-first model.
    """
    prompt_text = system_prompt
    if not prompt_text and system_prompt_file:
        target = Path(system_prompt_file)
        if target.exists() and target.is_file():
            prompt_text = target.read_text(encoding="utf-8").strip()

    if not prompt_text:
        raise ValueError("system_prompt or system_prompt_file is required.")

    ground_truth_text = ground_truth
    if not ground_truth_text and ground_truth_file:
        target = Path(ground_truth_file)
        if target.exists() and target.is_file():
            ground_truth_text = target.read_text(encoding="utf-8").strip()

    logs_payload: Any = None
    if log_path:
        target = Path(log_path)
        if target.exists() and target.is_file():
            logs_payload = target.read_text(encoding="utf-8", errors="ignore")

    return build_case(
        system_prompt=prompt_text,
        user_input=user_input,
        ground_truth=ground_truth_text,
        logs=logs_payload,
        workspace=workspace,
    )
