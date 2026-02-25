from __future__ import annotations

import argparse
import os
from typing import Any

from copilot_prompt_refiner.ingest.copilot import build_case_from_payload
from copilot_prompt_refiner.pipeline import PromptRefinementPipeline


def _as_bool(value: Any, default: bool = True) -> bool:
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


def _context_prompt_sources(context: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(context, dict):
        return []

    current_prompts = context.get("current_system_prompts")
    if not isinstance(current_prompts, dict):
        return []

    sources: list[dict[str, str]] = []
    for agent_name, prompt_text in current_prompts.items():
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue
        name = str(agent_name).strip() or "agent"
        escaped = prompt_text.replace('"""', '\\"""').strip()
        sources.append(
            {
                "path": f"context/{name}/definition.py",
                "content": f'SYSTEM_PROMPT = """{escaped}"""',
            }
        )
    return sources


def _context_system_prompt(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None

    direct_prompt = context.get("system_prompt")
    if isinstance(direct_prompt, str) and direct_prompt.strip():
        return direct_prompt.strip()

    current_prompts = context.get("current_system_prompts")
    if not isinstance(current_prompts, dict):
        return None

    sections: list[str] = []
    for agent_name, prompt_text in current_prompts.items():
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue
        name = str(agent_name).strip() or "agent"
        sections.append(f"## {name}\n{prompt_text.strip()}")

    if not sections:
        return None

    return "You are refining a multi-agent system prompt set.\n\n" + "\n\n".join(sections)


def _merge_prompt_sources(
    primary: Any,
    extras: list[dict[str, str]],
) -> Any:
    if not extras:
        return primary
    if primary is None:
        return extras
    if isinstance(primary, list):
        return [*primary, *extras]
    return [primary, *extras]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-refiner-mcp",
        description="Run Copilot Prompt Refiner MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.getenv("PROMPT_REFINER_MCP_TRANSPORT", "stdio"),
        help="MCP transport (stdio for local tool runner, streamable-http/sse for remote).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("PROMPT_REFINER_MCP_HOST", "127.0.0.1"),
        help="Host for HTTP/SSE transports.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PROMPT_REFINER_MCP_PORT", "8000")),
        help="Port for HTTP/SSE transports.",
    )
    parser.add_argument(
        "--mount-path",
        default=os.getenv("PROMPT_REFINER_MCP_MOUNT_PATH", "/"),
        help="Mount path used by SSE transport.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default=os.getenv("PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="Streamable HTTP endpoint path.",
    )
    parser.add_argument(
        "--sse-path",
        default=os.getenv("PROMPT_REFINER_MCP_SSE_PATH", "/sse"),
        help="SSE endpoint path.",
    )
    parser.add_argument(
        "--message-path",
        default=os.getenv("PROMPT_REFINER_MCP_MESSAGE_PATH", "/messages/"),
        help="SSE message endpoint path.",
    )
    parser.add_argument(
        "--stateless-http",
        default=os.getenv("PROMPT_REFINER_MCP_STATELESS_HTTP", "true"),
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        help=(
            "Use stateless Streamable HTTP mode (no session id required). "
            "Recommended for broad MCP client compatibility."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("PROMPT_REFINER_MCP_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="MCP server log level.",
    )
    return parser


def _to_payload_case(payload: dict[str, Any] | None):
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload_input must be a JSON object.")

    # Some MCP clients wrap tool args as {"payload_input": {...}} even when
    # the tool parameter is already named payload_input. Accept one extra layer.
    nested_payload = payload.get("payload_input")
    if isinstance(nested_payload, dict):
        payload = nested_payload

    context_files = payload.get("context_files")
    if context_files is not None and not isinstance(context_files, list):
        raise ValueError("context_files must be a list of strings.")

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object.")

    context = payload.get("context")
    if context is not None and not isinstance(context, dict):
        raise ValueError("context must be an object when provided.")
    context_obj: dict[str, Any] | None = context if isinstance(context, dict) else None

    prompt_sources = payload.get("prompt_sources")
    if prompt_sources is None:
        prompt_sources = payload.get("files")
    if prompt_sources is None and context_obj is not None:
        prompt_sources = context_obj.get("prompt_sources") or context_obj.get("files")
    prompt_sources = _merge_prompt_sources(
        primary=prompt_sources,
        extras=_context_prompt_sources(context_obj),
    )

    log_sources = payload.get("log_sources")
    if log_sources is None:
        log_sources = payload.get("logs_files")
    if log_sources is None and context_obj is not None:
        log_sources = context_obj.get("log_sources") or context_obj.get("logs_files")

    user_input = payload.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        alt_user_input = payload.get("copilot_user_input")
        if isinstance(alt_user_input, str):
            user_input = alt_user_input
        elif context_obj is not None and isinstance(context_obj.get("user_input"), str):
            user_input = context_obj.get("user_input")
        else:
            user_input = None

    system_prompt = payload.get("system_prompt")
    if not (isinstance(system_prompt, str) and system_prompt.strip()):
        system_prompt = _context_system_prompt(context_obj)

    definition_py_content = payload.get("definition_py_content")
    if not (
        isinstance(definition_py_content, str) and definition_py_content.strip()
    ) and context_obj is not None:
        maybe_definition = context_obj.get("definition_py_content")
        if isinstance(maybe_definition, str):
            definition_py_content = maybe_definition

    logs = payload.get("logs")
    if logs is None and context_obj is not None:
        logs = context_obj.get("logs")

    ground_truth = payload.get("ground_truth")
    if ground_truth is None and context_obj is not None:
        ground_truth = context_obj.get("ground_truth")

    ground_truth_content = payload.get("ground_truth_content")
    if ground_truth_content is None and context_obj is not None:
        ground_truth_content = context_obj.get("ground_truth_content")

    return build_case_from_payload(
        workspace=payload.get("workspace") or ".",
        system_prompt=system_prompt,
        definition_py_content=definition_py_content,
        prompt_sources=prompt_sources,
        user_input=user_input,
        require_user_input=_as_bool(payload.get("require_user_input"), True),
        ground_truth=ground_truth,
        ground_truth_content=ground_truth_content,
        logs=logs,
        log_sources=log_sources,
        context_files=context_files,
        case_id=payload.get("case_id"),
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "MCP SDK is not installed. Install with: pip install -e .[mcp]"
        ) from exc

    pipeline = PromptRefinementPipeline.default()
    mcp = FastMCP(
        "copilot-prompt-refiner",
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        sse_path=args.sse_path,
        message_path=args.message_path,
        streamable_http_path=args.streamable_http_path,
        stateless_http=_as_bool(args.stateless_http, True),
        log_level=args.log_level,
    )

    @mcp.tool()
    def discover_case_input(payload_input: dict[str, Any] | None = None) -> dict[str, Any]:
        case = _to_payload_case(payload_input)
        return {
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

    @mcp.tool()
    def evaluate_prompt(payload_input: dict[str, Any] | None = None) -> dict[str, Any]:
        case = _to_payload_case(payload_input)
        result = pipeline.evaluate(case)
        return result.to_dict()

    @mcp.tool()
    def refine_prompt(payload_input: dict[str, Any] | None = None) -> dict[str, Any]:
        case = _to_payload_case(payload_input)
        result = pipeline.refine(case)
        return result.to_dict()

    @mcp.tool()
    def run_refinement_pipeline(
        payload_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        case = _to_payload_case(payload_input)
        result = pipeline.run(
            case,
            max_iters=_as_int((payload_input or {}).get("max_iters")),
        )
        return result.to_dict()

    mcp.run(
        transport=args.transport,
        mount_path=args.mount_path if args.transport == "sse" else None,
    )


if __name__ == "__main__":
    main()
