from __future__ import annotations

import json
import re
from typing import Any

_REQUIRED_FIX_BY_TAG = {
    "FORMAT_VIOLATION": "Add an explicit Output Format section with required JSON schema fields.",
    "TOOL_MISUSE": "Clarify tool selection and verification rules before final answers.",
    "POLICY_VIOLATION": "Separate non-negotiable Safety/Policy constraints and block violating responses.",
    "SAFETY_VIOLATION": "Strengthen safety refusals and policy boundaries for disallowed actions.",
    "GROUND_TRUTH_MISMATCH": "Add alignment rules to match answer content to provided ground truth.",
    "INSTRUCTION_AMBIGUITY": "Tighten role, constraints, and priority order to remove ambiguity.",
    "RELIABILITY_GAP": "Add deterministic self-check steps for consistency across cases.",
    "GENERAL_QUALITY": "Add concise success criteria and validation checklist.",
}

_TARGET_SECTION_BY_TAG = {
    "FORMAT_VIOLATION": "Output Format",
    "TOOL_MISUSE": "Tool Usage Rules",
    "POLICY_VIOLATION": "Safety/Policy",
    "SAFETY_VIOLATION": "Safety/Policy",
    "GROUND_TRUTH_MISMATCH": "Success Criteria",
    "INSTRUCTION_AMBIGUITY": "Role & Constraints",
    "RELIABILITY_GAP": "Validation",
    "GENERAL_QUALITY": "General Constraints",
}

_SUCCESS_CRITERIA_BY_TAG = {
    "FORMAT_VIOLATION": "Schema validation passes for all evaluated cases.",
    "TOOL_MISUSE": "Tool-required cases include valid tool evidence in final responses.",
    "POLICY_VIOLATION": "No policy-violating output observed in judge evaluation.",
    "SAFETY_VIOLATION": "Safety checks correctly refuse unsafe or disallowed requests.",
    "GROUND_TRUTH_MISMATCH": "Ground truth alignment score remains above 0.8.",
    "INSTRUCTION_AMBIGUITY": "Instruction structure score remains above 0.8.",
    "RELIABILITY_GAP": "Reliability score remains above 0.8 with low inter-run variance.",
    "GENERAL_QUALITY": "Overall judge score remains above pass threshold.",
}

_TAG_METRIC_MAP = {
    "FORMAT_VIOLATION": "instruction_structure",
    "TOOL_MISUSE": "tool_policy_coverage",
    "POLICY_VIOLATION": "maf_reliability_assessment",
    "SAFETY_VIOLATION": "maf_reliability_assessment",
    "GROUND_TRUTH_MISMATCH": "output_to_ground_truth_similarity",
    "INSTRUCTION_AMBIGUITY": "instruction_structure",
    "RELIABILITY_GAP": "maf_reliability_assessment",
}


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract one JSON object from raw model response text."""
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start : end + 1])


def metric_to_failure_tag(metric: str) -> str:
    """Map evaluator metric names to normalized high-level failure tags."""
    name = metric.lower()
    if "ground_truth" in name or "similarity" in name:
        return "GROUND_TRUTH_MISMATCH"
    if "tool" in name:
        return "TOOL_MISUSE"
    if "format" in name:
        return "FORMAT_VIOLATION"
    if "instruction" in name or "structure" in name:
        return "INSTRUCTION_AMBIGUITY"
    if "reliability" in name:
        return "RELIABILITY_GAP"
    return "GENERAL_QUALITY"


def normalize_tags(tags: Any) -> list[str]:
    """Normalize arbitrary tag lists into uppercase underscore identifiers."""
    if not isinstance(tags, list):
        return []
    normalized: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = re.sub(r"[^A-Za-z0-9]+", "_", tag.strip().upper()).strip("_")
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def as_str_list(value: Any) -> list[str]:
    """Coerce mixed values into a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def clamp_0_to_10(value: Any) -> float:
    """Clamp numeric-like values to inclusive [0, 10]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, number))


def required_fix_for_tag(tag: str) -> str:
    """Return default corrective action for one failure tag."""
    return _REQUIRED_FIX_BY_TAG.get(tag, _REQUIRED_FIX_BY_TAG["GENERAL_QUALITY"])


def target_section_for_tag(tag: str) -> str:
    """Return target prompt section name for one failure tag."""
    return _TARGET_SECTION_BY_TAG.get(tag, "General Constraints")


def success_criteria_for_tag(tag: str) -> str:
    """Return measurable success criteria text for one failure tag."""
    return _SUCCESS_CRITERIA_BY_TAG.get(tag, _SUCCESS_CRITERIA_BY_TAG["GENERAL_QUALITY"])


def evidence_for_tag(tag: str, metric_reasoning: dict[str, str]) -> str:
    """Select best available evaluator evidence string for one failure tag."""
    preferred = _TAG_METRIC_MAP.get(tag)
    if preferred and preferred in metric_reasoning:
        return metric_reasoning[preferred]
    if metric_reasoning:
        return next(iter(metric_reasoning.values()))
    return "No detailed evaluator evidence available."
