import json

import pytest

from copilot_prompt_refiner.ingest.copilot import build_case_from_payload


def test_build_case_from_payload_parses_definition_logs_and_ground_truth() -> None:
    definition_content = 'SYSTEM_PROMPT = """You are an external agent. Use tools."""\n'
    logs_content = json.dumps(
        [
            {"role": "user", "content": "서비스 장애 원인을 알려줘"},
            {"role": "assistant", "content": "원인 분석 중"},
        ]
    )
    ground_truth_content = json.dumps(
        {"ground_truth": "DB migration 누락으로 배포 실패"}
    )

    case = build_case_from_payload(
        workspace="remote-repo",
        definition_py_content=definition_content,
        user_input="서비스 장애 원인을 알려줘",
        logs=logs_content,
        ground_truth_content=ground_truth_content,
        context_files=["definition.py", "logs/agent.json"],
    )

    assert case.system_prompt.startswith("You are an external agent")
    assert case.user_input == "서비스 장애 원인을 알려줘"
    assert case.ground_truth == "DB migration 누락으로 배포 실패"
    assert len(case.logs) == 2
    assert case.metadata.get("discovery", {}).get("mode") == "payload"


def test_build_case_from_payload_requires_user_input_or_user_logs() -> None:
    definition_content = 'SYSTEM_PROMPT = "You are an external agent."\n'

    with pytest.raises(ValueError):
        build_case_from_payload(
            workspace="remote-repo",
            definition_py_content=definition_content,
            logs=[{"role": "assistant", "content": "응답만 있음"}],
        )


def test_build_case_from_payload_selects_prompt_from_multiple_sources() -> None:
    prompt_sources = [
        {
            "path": "agents/router.py",
            "content": 'AGENT_PROMPT = "short"',
        },
        {
            "path": "agents/reviewer.ts",
            "content": 'export const systemPrompt = `You are a reviewer agent. Validate outputs and constraints.`',
        },
        {
            "path": "multi_agent/definition.py",
            "content": 'SYSTEM_PROMPT = """You are the orchestrator agent. Coordinate specialized sub-agents and enforce tool policy."""',
        },
    ]

    case = build_case_from_payload(
        workspace="remote-repo",
        prompt_sources=prompt_sources,
        user_input="실패한 워크플로우 원인 분석해줘",
        logs=[{"role": "user", "content": "실패한 워크플로우 원인 분석해줘"}],
    )

    assert "orchestrator agent" in case.system_prompt
    discovery = case.metadata.get("discovery", {})
    assert str(discovery.get("system_prompt_source", "")).startswith("prompt_sources:")
    assert int(discovery.get("system_prompt_candidates", 0)) >= 2


def test_build_case_from_payload_picks_latest_log_source_for_inference() -> None:
    case = build_case_from_payload(
        workspace="remote-repo",
        definition_py_content='SYSTEM_PROMPT = "You are an external agent."\n',
        require_user_input=False,
        log_sources=[
            {
                "path": "logs/old.json",
                "modified_at": "2026-01-01T00:00:00Z",
                "content": '[{"role":"user","content":"오래된 입력"}]',
            },
            {
                "path": "logs/new.json",
                "modified_at": "2026-02-01T00:00:00Z",
                "content": '[{"role":"user","content":"최신 입력"}]',
            },
        ],
    )

    assert case.user_input == "최신 입력"
    discovery = case.metadata.get("discovery", {})
    assert str(discovery.get("log_source", "")).endswith("logs/new.json")
