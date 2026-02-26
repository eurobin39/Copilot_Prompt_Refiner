from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageLog:
    """Normalized conversation/log record used across ingestion and evaluation.

    Stores role/content plus optional timestamp and extra metadata fields from
    upstream clients or tool events.
    """

    role: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this dataclass to a JSON-compatible dictionary.

        Used by MCP/CLI response layers to emit stable structured artifacts.
        """
        return asdict(self)


@dataclass(slots=True)
class AgentCase:
    """Complete evaluation input bundle for one prompt assessment case.

    It contains prompt/user/ground-truth text with logs and metadata required
    by evaluators, JudgeAgent, and refinement workflows.
    """

    case_id: str
    system_prompt: str
    user_input: str
    ground_truth: str | None = None
    logs: list[MessageLog] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize case fields for transport and debugging output.

        Keeps nested dataclass structures in plain dict/list form for JSON use.
        """
        return asdict(self)


@dataclass(slots=True)
class EvaluatorScore:
    """Per-evaluator metric result consumed by Judge aggregation.

    Includes weighted score and short reasoning to preserve traceability from
    aggregate verdicts back to individual metric checks.
    """

    evaluator: str
    metric: str
    score: float
    weight: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluator score for logging and API responses.

        This keeps metric output portable across CLI, MCP, and test fixtures.
        """
        return asdict(self)


@dataclass(slots=True)
class ModelReview:
    """Structured review emitted by one reviewer model or heuristic fallback.

    Captures 0-10 score, violation flags, failure tags, and improvement hints
    used for consensus and prioritized action generation.
    """

    reviewer_model: str
    score_0_to_10: float
    rule_violation: bool = False
    failure_tags: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize model review into JSON-compatible primitive types.

        Useful when returning per-model evidence to clients and trace logs.
        """
        return asdict(self)


@dataclass(slots=True)
class FailureCase:
    """Concrete failure record suitable for repro and repair planning.

    Each entry ties a failure type to evidence, reproduction context, and
    explicit success criteria for subsequent refinement validation.
    """

    case_id: str
    failure_type: str
    evidence: str
    reproduction_input: str
    reproduction_output: str
    required_fix: str
    success_criteria: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize failure case for downstream reporting and tooling.

        Preserves all reproducibility fields in a transport-safe shape.
        """
        return asdict(self)


@dataclass(slots=True)
class PrioritizedAction:
    """Ordered prompt-fix action derived from Judge consensus.

    Actions include target section and rationale so Refine can apply focused
    edits instead of unbounded prompt rewrites.
    """

    priority: int
    action: str
    target_section: str
    rationale: str
    linked_failure_type: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize prioritized action to a plain dictionary.

        Enables deterministic action transfer between Judge and response layers.
        """
        return asdict(self)


@dataclass(slots=True)
class IterationTrace:
    """Lightweight per-iteration telemetry for pipeline convergence analysis.

    Tracks score, pass state, policy/format indicators, and top actions chosen
    at each refinement step.
    """

    iteration: int
    score: float
    passed: bool
    policy_violation: bool
    format_pass_rate: float
    top_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize iteration trace entry for final pipeline outputs.

        Used by clients to inspect refinement progress over multiple iterations.
        """
        return asdict(self)


@dataclass(slots=True)
class JudgeResult:
    """Canonical Judge verdict object consumed by Refine and clients.

    Combines aggregate decision fields with rich supporting evidence such as
    evaluator scores, model reviews, failure cases, and prioritized actions.
    """

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
        """Serialize Judge verdict into JSON-ready nested dictionaries/lists.

        This is the payload shape returned by MCP and CLI evaluate endpoints.
        """
        return asdict(self)


@dataclass(slots=True)
class PromptRevision:
    """Result of applying Judge-guided edits to a system prompt.

    Includes revised prompt text, applied actions, targeted failures, growth
    ratio, and human-readable patch/change notes.
    """

    original_prompt: str
    refined_prompt: str
    change_log: list[str] = field(default_factory=list)
    patch_notes: list[str] = field(default_factory=list)
    applied_actions: list[str] = field(default_factory=list)
    targeted_failures: list[str] = field(default_factory=list)
    prompt_growth_ratio: float = 0.0
    confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        """Serialize prompt revision for transport to external callers.

        Keeps refinement artifacts explicit for auditing and follow-up runs.
        """
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    """Full output of iterative refinement pipeline execution.

    Bundles final case/judge/revision artifacts with iteration count, stop
    reason, and trace timeline.
    """

    case: AgentCase
    judge: JudgeResult
    revision: PromptRevision
    iterations: int = 1
    stopped_reason: str = "completed"
    traces: list[IterationTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full pipeline result to a nested dictionary.

        Intended for MCP responses and persisted debugging snapshots.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize full pipeline result as pretty-printed JSON text.

        Primarily used by CLI and ad-hoc local inspection workflows.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
