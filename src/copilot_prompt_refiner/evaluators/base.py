from __future__ import annotations

from abc import ABC, abstractmethod

from copilot_prompt_refiner.models import AgentCase, EvaluatorScore


class Evaluator(ABC):
    name: str = "base"
    metric: str = "unknown"
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, case: AgentCase, candidate_prompt: str) -> EvaluatorScore:
        raise NotImplementedError
