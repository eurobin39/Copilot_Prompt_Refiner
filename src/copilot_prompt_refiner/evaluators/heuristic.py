from __future__ import annotations

import re

from copilot_prompt_refiner.evaluators.base import Evaluator
from copilot_prompt_refiner.models import AgentCase, EvaluatorScore


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) > 2
    }


def _jaccard(a: str, b: str) -> float:
    left = _tokenize(a)
    right = _tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class PromptStructureEvaluator(Evaluator):
    name = "prompt_structure"
    metric = "instruction_structure"
    weight = 1.0

    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        checks = {
            "role": ["you are", "role", "agent"],
            "constraints": ["must", "must not", "do not", "never"],
            "tools": ["tool", "mcp", "function", "call"],
            "output_format": ["output", "format", "json", "markdown"],
            "validation": ["check", "verify", "validate"],
        }

        normalized = candidate_prompt.lower()
        hit_count = 0
        missing: list[str] = []
        for section, words in checks.items():
            if any(word in normalized for word in words):
                hit_count += 1
            else:
                missing.append(section)

        score = hit_count / len(checks)
        reason = (
            "Prompt structure check complete. "
            f"Missing sections: {', '.join(missing) if missing else 'none'}."
        )

        return EvaluatorScore(
            evaluator=self.name,
            metric=self.metric,
            score=score,
            weight=self.weight,
            reasoning=reason,
        )


class GroundTruthAlignmentEvaluator(Evaluator):
    name = "ground_truth_alignment"
    metric = "output_to_ground_truth_similarity"
    weight = 1.2

    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        if not case.ground_truth:
            return EvaluatorScore(
                evaluator=self.name,
                metric=self.metric,
                score=0.5,
                weight=self.weight,
                reasoning="No ground truth provided; fallback neutral score applied.",
            )

        last_agent_output = ""
        for message in reversed(case.logs):
            if message.role.lower() in {"assistant", "agent"}:
                last_agent_output = message.content
                break

        if not last_agent_output:
            return EvaluatorScore(
                evaluator=self.name,
                metric=self.metric,
                score=0.35,
                weight=self.weight,
                reasoning="Ground truth exists but no assistant output in logs.",
            )

        similarity = _jaccard(last_agent_output, case.ground_truth)
        reason = f"Assistant output-ground truth token overlap: {similarity:.2f}."
        return EvaluatorScore(
            evaluator=self.name,
            metric=self.metric,
            score=min(1.0, max(0.0, similarity)),
            weight=self.weight,
            reasoning=reason,
        )


class ToolUsageEvaluator(Evaluator):
    name = "tool_usage"
    metric = "tool_policy_coverage"
    weight = 0.8

    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        prompt_mentions_tools = any(
            key in candidate_prompt.lower() for key in ["tool", "mcp", "function", "api"]
        )
        tool_calls = [m for m in case.logs if m.role.lower() == "tool"]

        if tool_calls and not prompt_mentions_tools:
            score = 0.3
            reason = "Logs include tool calls but prompt lacks explicit tool policy guidance."
        elif prompt_mentions_tools and tool_calls:
            score = 0.9
            reason = "Prompt and logs both indicate tool-awareness and usage."
        elif prompt_mentions_tools:
            score = 0.75
            reason = "Prompt includes tool policy but no tool-call evidence in logs."
        else:
            score = 0.5
            reason = "No explicit tool policy detected and no tool calls observed."

        return EvaluatorScore(
            evaluator=self.name,
            metric=self.metric,
            score=score,
            weight=self.weight,
            reasoning=reason,
        )
