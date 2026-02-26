from __future__ import annotations

from copilot_prompt_refiner.agents.microsoft_agent_framework import (
    MicrosoftAgentFrameworkRuntime,
)
from copilot_prompt_refiner.agents.judge_agent import JudgeAgent
from copilot_prompt_refiner.agents.refine_agent import RefineAgent
from copilot_prompt_refiner.config import RuntimeConfig, load_dotenv
from copilot_prompt_refiner.evaluators.heuristic import (
    GroundTruthAlignmentEvaluator,
    PromptStructureEvaluator,
    ToolUsageEvaluator,
)
from copilot_prompt_refiner.evaluators.maf_judge import MAFJudgeEvaluator
from copilot_prompt_refiner.models import (
    AgentCase,
    IterationTrace,
    JudgeResult,
    PipelineResult,
    PromptRevision,
)


class PromptRefinementPipeline:
    def __init__(
        self,
        judge_agent: JudgeAgent,
        refine_agent: RefineAgent,
        max_iters: int = 3,
    ) -> None:
        self.judge_agent = judge_agent
        self.refine_agent = refine_agent
        self.max_iters = max(1, max_iters)

    @classmethod
    def default(
        cls,
        pass_threshold: float | None = None,
        use_maf: bool | None = None,
        strict_maf: bool | None = None,
        max_iters: int | None = None,
    ) -> "PromptRefinementPipeline":
        """Build a pipeline from env/config defaults with optional runtime overrides."""
        load_dotenv()
        config = RuntimeConfig.from_env()

        resolved_threshold = (
            pass_threshold if pass_threshold is not None else config.default_pass_threshold
        )
        resolved_use_maf = (
            use_maf if use_maf is not None else config.use_microsoft_agent_framework
        )
        resolved_strict_maf = (
            strict_maf
            if strict_maf is not None
            else config.strict_microsoft_agent_framework
        )
        resolved_max_iters = (
            max_iters if max_iters is not None else config.default_max_iters
        )

        evaluators = [
            PromptStructureEvaluator(),
            GroundTruthAlignmentEvaluator(),
            ToolUsageEvaluator(),
        ]

        runtime: MicrosoftAgentFrameworkRuntime | None = None
        refine = RefineAgent(
            strict_runtime=resolved_strict_maf,
            max_prompt_growth_ratio=config.refine_max_prompt_growth_ratio,
            max_actions=config.refine_max_actions,
        )
        if resolved_use_maf:
            runtime = MicrosoftAgentFrameworkRuntime(model=config.default_model)
            if resolved_strict_maf:
                runtime.validate_configuration()

            evaluators.append(
                MAFJudgeEvaluator(runtime=runtime, strict_runtime=resolved_strict_maf)
            )
            refine = RefineAgent(
                runtime=runtime,
                strict_runtime=resolved_strict_maf,
                max_prompt_growth_ratio=config.refine_max_prompt_growth_ratio,
                max_actions=config.refine_max_actions,
            )

        judge = JudgeAgent(
            evaluators=evaluators,
            pass_threshold=resolved_threshold,
            runtime=runtime,
            strict_runtime=resolved_strict_maf,
            review_models=config.judge_review_models,
            disagreement_threshold=config.judge_disagreement_threshold,
        )
        return cls(judge_agent=judge, refine_agent=refine, max_iters=resolved_max_iters)

    def evaluate(self, case: AgentCase, candidate_prompt: str | None = None) -> JudgeResult:
        """Score a prompt candidate and return a structured Judge verdict."""
        return self.judge_agent.judge(case, candidate_prompt=candidate_prompt)

    def refine(self, case: AgentCase) -> PromptRevision:
        """Run one-pass judge+refine and return only the prompt revision."""
        judge_result = self.evaluate(case)
        return self.refine_agent.refine(case, judge_result)

    def run(self, case: AgentCase, max_iters: int | None = None) -> PipelineResult:
        """Iterate evaluate/refine until pass threshold or max iteration budget is reached."""
        iter_limit = max(1, max_iters if max_iters is not None else self.max_iters)
        current_prompt = case.system_prompt
        traces: list[IterationTrace] = []
        latest_revision = self._identity_revision(current_prompt)

        for iteration in range(1, iter_limit + 1):
            iter_case = self._case_with_prompt(case, current_prompt)
            judge_result = self.evaluate(iter_case, candidate_prompt=current_prompt)
            traces.append(
                IterationTrace(
                    iteration=iteration,
                    score=judge_result.overall_score,
                    passed=judge_result.passed,
                    policy_violation=judge_result.policy_violation,
                    format_pass_rate=judge_result.format_pass_rate,
                    top_actions=judge_result.recommended_actions[:3],
                )
            )

            if judge_result.passed:
                return PipelineResult(
                    case=iter_case,
                    judge=judge_result,
                    revision=latest_revision,
                    iterations=iteration,
                    stopped_reason="threshold_reached",
                    traces=traces,
                )

            if iteration >= iter_limit:
                judge_result.feedback.append(
                    self.judge_agent.summarize_max_iters(
                        final_judge=judge_result,
                        max_iters=iter_limit,
                    )
                )
                return PipelineResult(
                    case=iter_case,
                    judge=judge_result,
                    revision=latest_revision,
                    iterations=iteration,
                    stopped_reason="max_iters_reached",
                    traces=traces,
                )

            latest_revision = self.refine_agent.refine(iter_case, judge_result)
            current_prompt = latest_revision.refined_prompt

        unreachable_case = self._case_with_prompt(case, current_prompt)
        fallback_judge = self.evaluate(unreachable_case, candidate_prompt=current_prompt)
        return PipelineResult(
            case=unreachable_case,
            judge=fallback_judge,
            revision=latest_revision,
            iterations=iter_limit,
            stopped_reason="completed",
            traces=traces,
        )

    def _case_with_prompt(self, case: AgentCase, prompt: str) -> AgentCase:
        """Clone a case while swapping only the candidate system prompt."""
        return AgentCase(
            case_id=case.case_id,
            system_prompt=prompt,
            user_input=case.user_input,
            ground_truth=case.ground_truth,
            logs=list(case.logs),
            context_files=list(case.context_files),
            metadata=dict(case.metadata),
        )

    def _identity_revision(self, prompt: str) -> PromptRevision:
        """Return a no-op revision used before any refinement is applied."""
        return PromptRevision(
            original_prompt=prompt,
            refined_prompt=prompt,
            change_log=["No refinement applied."],
            patch_notes=["No refinement applied."],
            applied_actions=[],
            targeted_failures=[],
            prompt_growth_ratio=0.0,
            confidence=0.9,
        )
