from __future__ import annotations

import re

from copilot_prompt_refiner.agents.base import TextGenerationRuntime
from copilot_prompt_refiner.evaluators.base import Evaluator
from copilot_prompt_refiner.models import AgentCase, EvaluatorScore


def _extract_score(text: str) -> float:
    """Extract a normalized `[0, 1]` score from free-form judge text.

    The parser prioritizes explicit `SCORE:` fields and falls back to first
    plausible numeric token to tolerate minor format drift.
    """
    # Prefer explicit SCORE line, otherwise first numeric token in [0,1].
    match = re.search(r"score\s*[:=]\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if not match:
        return 0.5

    try:
        value = float(match.group(1))
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, value))


class MAFJudgeEvaluator(Evaluator):
    """Runtime-backed evaluator that asks an LLM to assess prompt reliability.

    This complements heuristic evaluators with broader judgment over safety,
    instruction-following, and agent workflow quality.
    """

    name = "maf_judge"
    metric = "maf_reliability_assessment"
    weight = 1.4

    def __init__(self, runtime: TextGenerationRuntime, strict_runtime: bool = True) -> None:
        """Initialize evaluator with runtime and strictness policy.

        In strict mode, runtime failures surface immediately; otherwise a
        neutral fallback score is returned with diagnostic reasoning.
        """
        self.runtime = runtime
        self.strict_runtime = strict_runtime

    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        """Request LLM judgment for the candidate prompt and normalize its output.

        The evaluator enforces a compact response format so Judge aggregation can
        consume score and rationale without additional post-processing complexity.
        """
        assistant_output = ""
        for message in reversed(case.logs):
            if message.role.lower() in {"assistant", "agent"}:
                assistant_output = message.content
                break

        user_instruction = (
            "Evaluate system prompt quality for an agentic workflow.\n"
            "Return strict format:\n"
            "SCORE: <0.00-1.00>\n"
            "REASON: <one sentence>\n\n"
            f"System Prompt:\n{candidate_prompt}\n\n"
            f"User Input:\n{case.user_input}\n\n"
            f"Ground Truth:\n{case.ground_truth or 'N/A'}\n\n"
            f"Assistant Output:\n{assistant_output or 'N/A'}\n"
        )

        try:
            raw = self.runtime.complete(
                system_prompt=(
                    "You are a strict Judge Agent for Microsoft Agent Framework workflows. "
                    "Score reliability, instruction-following, tool-use policy clarity, and "
                    "alignment to ground truth when available."
                ),
                user_input=user_instruction,
            )
        except Exception as exc:
            if self.strict_runtime:
                raise
            return EvaluatorScore(
                evaluator=self.name,
                metric=self.metric,
                score=0.5,
                weight=self.weight,
                reasoning=f"MAF judge unavailable: {exc}",
            )

        score = _extract_score(raw)
        reason_match = re.search(r"reason\s*[:=]\s*(.+)", raw, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else raw.strip()[:220]

        return EvaluatorScore(
            evaluator=self.name,
            metric=self.metric,
            score=score,
            weight=self.weight,
            reasoning=reason or "MAF judge response parsed without explicit reason.",
        )
