from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageLog:
    role: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentCase:
    case_id: str
    system_prompt: str
    user_input: str
    ground_truth: str | None = None
    logs: list[MessageLog] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluatorScore:
    evaluator: str
    metric: str
    score: float
    weight: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelReview:
    reviewer_model: str
    score_0_to_10: float
    rule_violation: bool = False
    failure_tags: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureCase:
    case_id: str
    failure_type: str
    evidence: str
    reproduction_input: str
    reproduction_output: str
    required_fix: str
    success_criteria: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PrioritizedAction:
    priority: int
    action: str
    target_section: str
    rationale: str
    linked_failure_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IterationTrace:
    iteration: int
    score: float
    passed: bool
    policy_violation: bool
    format_pass_rate: float
    top_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JudgeResult:
    overall_score: float
    pass_threshold: float = 0.72
    passed: bool = False
    feedback: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    evaluator_scores: list[EvaluatorScore] = field(default_factory=list)
    policy_violation: bool = False
    format_pass_rate: float = 1.0
    disagreement_flag: bool = False
    execution_summary: dict[str, Any] = field(default_factory=dict)
    aggregation_summary: dict[str, Any] = field(default_factory=dict)
    per_model_reviews: list[ModelReview] = field(default_factory=list)
    failure_cases: list[FailureCase] = field(default_factory=list)
    prioritized_actions: list[PrioritizedAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptRevision:
    original_prompt: str
    refined_prompt: str
    change_log: list[str] = field(default_factory=list)
    patch_notes: list[str] = field(default_factory=list)
    applied_actions: list[str] = field(default_factory=list)
    targeted_failures: list[str] = field(default_factory=list)
    prompt_growth_ratio: float = 0.0
    confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    case: AgentCase
    judge: JudgeResult
    revision: PromptRevision
    iterations: int = 1
    stopped_reason: str = "completed"
    traces: list[IterationTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
