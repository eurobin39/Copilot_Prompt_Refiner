from copilot_prompt_refiner.mcp_server import _to_payload_case


def test_to_payload_case_supports_context_current_system_prompts() -> None:
    case = _to_payload_case(
        {
            "user_input": "Improve these prompts",
            "context": {
                "project": "Resume Assistant",
                "current_system_prompts": {
                    "resume_info_collector": "You are an expert Resume Information Collector.",
                    "resume_writer": "You are an expert Resume Writer.",
                },
            },
        }
    )

    assert "multi-agent system prompt set" in case.system_prompt
    assert "resume_info_collector" in case.system_prompt
    assert case.user_input == "Improve these prompts"


def test_to_payload_case_supports_context_user_input_fallback() -> None:
    case = _to_payload_case(
        {
            "context": {
                "user_input": "Context fallback input",
                "current_system_prompts": {
                    "resume_reviewer": "You are an expert Resume Reviewer."
                },
            }
        }
    )

    assert case.user_input == "Context fallback input"
    assert "resume_reviewer" in case.system_prompt


def test_to_payload_case_supports_nested_payload_input_wrapper() -> None:
    case = _to_payload_case(
        {
            "payload_input": {
                "user_input": "Nested wrapper input",
                "context": {
                    "current_system_prompts": {
                        "resume_writer": "You are an expert Resume Writer."
                    }
                },
            }
        }
    )

    assert case.user_input == "Nested wrapper input"
    assert "resume_writer" in case.system_prompt


def test_to_payload_case_auto_fills_empty_payload_shape_slots() -> None:
    case = _to_payload_case(
        {
            "system_prompt": "You are a prompt refiner.",
            "user_input": "Check payload shape",
        }
    )

    payload_shape = case.metadata.get("payload_shape", {})
    assert payload_shape.get("logs") == "auto_filled"
    assert payload_shape.get("log_sources") == "auto_filled"
    assert payload_shape.get("ground_truth_content") == "auto_filled"
