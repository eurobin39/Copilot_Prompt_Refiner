from __future__ import annotations

from abc import ABC, abstractmethod

from copilot_prompt_refiner.models import AgentCase, EvaluatorScore


class Evaluator(ABC):
    """Abstract contract for prompt-quality evaluators used by JudgeAgent.

    Implementations produce normalized `EvaluatorScore` objects so aggregation
    logic can combine heterogeneous metrics with consistent weighting.
    """

    name: str = "base"
    metric: str = "unknown"
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        """Score one candidate prompt against a case context.

        Returned score must be normalized to `[0, 1]` semantics and include
        reasoning text for traceability in Judge feedback.
        """
        raise NotImplementedError
