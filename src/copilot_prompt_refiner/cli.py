from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from copilot_prompt_refiner.ingest.copilot import build_case_from_payload
from copilot_prompt_refiner.pipeline import PromptRefinementPipeline


def _print_json(data: dict[str, Any]) -> None:
    """Print structured output as pretty JSON for CLI users.

    UTF-8 characters are preserved to keep logs and multilingual payload fields
    readable during local debugging.
    """
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _as_bool(value: Any, default: bool = True) -> bool:
    """Coerce loosely typed CLI payload values into boolean flags.

    This mirrors server-side parsing so behavior stays consistent between
    `prompt-refiner` CLI and MCP tool execution paths.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    """Coerce payload values into optional integer settings.

    Invalid values gracefully fall back, avoiding CLI crashes for malformed
    ad-hoc payloads.
    """
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default
    return default


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for discover/evaluate/refine/run subcommands.

    All subcommands share payload input options so users can switch between
    inspection and execution without changing invocation style.
    """
    parser = argparse.ArgumentParser(
        prog="prompt-refiner",
        description="Payload-first prompt refinement framework for external agents.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("discover", "evaluate", "refine", "run"):
        node = sub.add_parser(command, help=f"{command.title()} from payload input")
        node.add_argument("--payload-file", type=Path)
        node.add_argument("--payload-json")

    return parser


def _extract_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Unwrap payload when input is nested under `payload_input`.

    Some clients send wrapped request envelopes; this helper normalizes both
    wrapped and direct JSON objects to one shape.
    """
    payload = raw.get("payload_input")
    if isinstance(payload, dict):
        return payload
    return raw


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Load payload JSON from file path or inline JSON argument.

    Exactly one payload source is required so command behavior is explicit and
    reproducible.
    """
    if args.payload_file is None and not args.payload_json:
        raise ValueError("Provide --payload-file or --payload-json.")

    if args.payload_file is not None:
        text = args.payload_file.read_text(encoding="utf-8")
        return _extract_payload(json.loads(text))

    return _extract_payload(json.loads(args.payload_json))


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for payload-first prompt evaluation and refinement.

    Resolves payload into `AgentCase`, dispatches selected command, and emits
    JSON output compatible with automation and manual inspection.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = _load_payload(args)
    user_input = payload.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        alt_user_input = payload.get("copilot_user_input")
        if isinstance(alt_user_input, str):
            user_input = alt_user_input
        else:
            user_input = None

    log_sources = payload.get("log_sources")
    if log_sources is None:
        log_sources = payload.get("logs_files")

    case = build_case_from_payload(
        workspace=payload.get("workspace") or ".",
        system_prompt=payload.get("system_prompt"),
        definition_py_content=payload.get("definition_py_content"),
        prompt_sources=payload.get("prompt_sources")
        if payload.get("prompt_sources") is not None
        else payload.get("files"),
        user_input=user_input,
        require_user_input=_as_bool(payload.get("require_user_input"), True),
        ground_truth=payload.get("ground_truth"),
        ground_truth_content=payload.get("ground_truth_content"),
        logs=payload.get("logs"),
        log_sources=log_sources,
        context_files=payload.get("context_files"),
        case_id=payload.get("case_id"),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    )

    pipeline = PromptRefinementPipeline.default()

    if args.command == "discover":
        _print_json(
            {
                "case_input": {
                    "case_id": case.case_id,
                    "system_prompt": case.system_prompt,
                    "user_input": case.user_input,
                    "ground_truth": case.ground_truth,
                    "log_count": len(case.logs),
                    "context_files": case.context_files,
                },
                "metadata": case.metadata,
            }
        )
        return

    if args.command == "evaluate":
        _print_json(pipeline.evaluate(case).to_dict())
        return

    if args.command == "refine":
        _print_json(pipeline.refine(case).to_dict())
        return

    _print_json(
        pipeline.run(
            case,
            max_iters=_as_int(payload.get("max_iters")),
        ).to_dict()
    )


if __name__ == "__main__":
    main()
