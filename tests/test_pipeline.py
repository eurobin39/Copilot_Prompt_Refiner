from copilot_prompt_refiner.ingest.copilot import build_case
from copilot_prompt_refiner.models import JudgeResult
from copilot_prompt_refiner.pipeline import PromptRefinementPipeline
from copilot_prompt_refiner.agents.refine_agent import RefineAgent
from copilot_prompt_refiner.agents.microsoft_agent_framework import AzureOpenAIRequestError


def test_pipeline_runs_and_refines_prompt() -> None:
    case = build_case(
        system_prompt="You are a coding assistant.",
        user_input="Summarize this issue",
        ground_truth="Short and factual summary",
        workspace=".",
    )
    pipeline = PromptRefinementPipeline.default(use_maf=False, strict_maf=False)
    result = pipeline.run(case)

    assert 0.0 <= result.judge.overall_score <= 1.0
    assert result.revision.refined_prompt
    assert (
        "## Minimal Patch" in result.revision.refined_prompt
        or "## Tool Usage Rules" in result.revision.refined_prompt
        or "## Output Format" in result.revision.refined_prompt
        or result.revision.prompt_growth_ratio == 0.0
    )


def test_judge_returns_structured_outputs() -> None:
    case = build_case(
        system_prompt="You are an assistant.",
        user_input="Summarize this in JSON.",
        logs=[
            {"role": "user", "content": "Summarize this in JSON."},
            {"role": "assistant", "content": "Plain text response"},
        ],
        workspace=".",
    )
    pipeline = PromptRefinementPipeline.default(use_maf=False, strict_maf=False, max_iters=1)
    result = pipeline.evaluate(case)

    assert result.per_model_reviews
    assert result.prioritized_actions
    assert isinstance(result.failure_cases, list)


class _ContentFilterRuntime:
    def complete(self, *, system_prompt: str, user_input: str) -> str:
        raise AzureOpenAIRequestError(
            message="Azure OpenAI request failed: HTTP 400.",
            status_code=400,
            route_url="https://example.openai.azure.com/openai/v1/chat/completions",
            response_json={
                "error": {
                    "code": "content_filter",
                    "innererror": {
                        "code": "ResponsibleAIPolicyViolation",
                        "content_filter_result": {
                            "jailbreak": {"detected": True, "filtered": True}
                        },
                    },
                }
            },
        )


class _RetryOnFilterRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system_prompt: str, user_input: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise AzureOpenAIRequestError(
                message="Azure OpenAI request failed: HTTP 400.",
                status_code=400,
                route_url="https://example.openai.azure.com/openai/v1/chat/completions",
                response_json={
                    "error": {
                        "code": "content_filter",
                        "innererror": {
                            "code": "ResponsibleAIPolicyViolation",
                            "content_filter_result": {
                                "jailbreak": {"detected": True, "filtered": True}
                            },
                        },
                    }
                },
            )
        return "You are an assistant.\n\n## Output Format\n- Return strict JSON."


class _JsonRefineRuntime:
    def complete(self, *, system_prompt: str, user_input: str) -> str:
        return (
            '{"refined_prompt":"You are an assistant. Return strict JSON only.",'
            '"summary":"Applied format and validation fixes."}'
        )


def test_refine_falls_back_on_content_filter_even_in_strict_mode() -> None:
    case = build_case(
        system_prompt="You are an assistant.",
        user_input="Respond in JSON.",
        workspace=".",
    )
    judge = JudgeResult(
        overall_score=0.4,
        pass_threshold=0.72,
        passed=False,
        recommended_actions=["Add JSON output format section."],
    )
    agent = RefineAgent(runtime=_ContentFilterRuntime(), strict_runtime=True)
    revision = agent.refine(case, judge)

    assert revision.refined_prompt
    matched = [note for note in revision.patch_notes if "content filter" in note.lower()]
    assert len(matched) == 1
    assert any("categories=jailbreak(detected=true,filtered=true)" in note for note in revision.patch_notes)


def test_refine_retries_with_compact_prompt_on_content_filter() -> None:
    case = build_case(
        system_prompt="You are an assistant.",
        user_input="Respond in JSON.",
        workspace=".",
    )
    judge = JudgeResult(
        overall_score=0.4,
        pass_threshold=0.72,
        passed=False,
        recommended_actions=["Add JSON output format section."],
    )
    runtime = _RetryOnFilterRuntime()
    agent = RefineAgent(runtime=runtime, strict_runtime=True)
    revision = agent.refine(case, judge)

    assert runtime.calls == 2
    assert "## Output Format" in revision.refined_prompt
    assert any("retried with ultra-compact context" in note.lower() for note in revision.patch_notes)
    assert any("code=content_filter" in note for note in revision.patch_notes)
    assert not any("used heuristic fallback patch" in note.lower() for note in revision.patch_notes)


def test_refine_parses_strict_json_runtime_response() -> None:
    case = build_case(
        system_prompt="You are an assistant.",
        user_input="Respond in JSON.",
        workspace=".",
    )
    judge = JudgeResult(
        overall_score=0.4,
        pass_threshold=0.72,
        passed=False,
        recommended_actions=["Add JSON output format section."],
    )
    agent = RefineAgent(runtime=_JsonRefineRuntime(), strict_runtime=True)
    revision = agent.refine(case, judge)

    assert revision.refined_prompt == "You are an assistant. Return strict JSON only."
    assert any("runtime summary: applied format and validation fixes." in note.lower() for note in revision.patch_notes)


def test_refine_skips_patch_when_growth_budget_is_too_small() -> None:
    case = build_case(
        system_prompt="Short prompt.",
        user_input="Respond in JSON.",
        workspace=".",
    )
    judge = JudgeResult(
        overall_score=0.2,
        pass_threshold=0.72,
        passed=False,
        recommended_actions=["Add an explicit Output Format section with required JSON schema fields."],
    )

    agent = RefineAgent(runtime=None, strict_runtime=False, max_prompt_growth_ratio=0.01)
    revision = agent.refine(case, judge)

    assert revision.refined_prompt == case.system_prompt
    assert revision.prompt_growth_ratio == 0.0
    assert any("growth budget is too small" in note.lower() for note in revision.patch_notes)
