from __future__ import annotations

RUNTIME_SYSTEM_PROMPT = "You are Refiner Agent for production prompts."
SAFE_RUNTIME_SYSTEM_PROMPT = "You are a concise prompt editor."

OUTPUT_FORMAT_PATCH = (
    "\n\n## Output Format\n"
    "- Always return valid JSON.\n"
    "- Include keys: decision, evidence, assumptions, next_actions.\n"
    "- If data is missing, set the field to null and explain why."
)

TOOL_USAGE_PATCH = (
    "\n\n## Tool Usage Rules\n"
    "- Use tools when factual verification is required.\n"
    "- Cite which tool output supports each critical claim.\n"
    "- If tool access fails, report limitation and fallback."
)

SAFETY_POLICY_PATCH = (
    "\n\n## Safety/Policy\n"
    "- Refuse requests that violate policy constraints.\n"
    "- Do not output restricted or unsafe instructions.\n"
    "- When refusing, provide a brief safe alternative."
)

SUCCESS_CRITERIA_PATCH = (
    "\n\n## Success Criteria\n"
    "- Align output with provided ground truth and constraints.\n"
    "- Avoid unsupported claims; tie conclusions to evidence."
)

VALIDATION_PATCH = (
    "\n\n## Validation\n"
    "- Before final answer, verify constraints, tool evidence, and format."
)


def build_runtime_instruction(
    *,
    current_prompt: str,
    low_scoring_categories: str,
    verdict_context: str,
    ground_truth: str | None,
    action_lines: str,
    growth_hint: str,
) -> str:
    """Render runtime refinement instruction with full context."""
    return (
        "You are Refiner Agent.\n"
        "Return strict JSON:\n"
        "{\n"
        '  "refined_prompt": "string",\n'
        '  "summary": "string"\n'
        "}\n"
        "Rules:\n"
        "- Preserve task intent.\n"
        "- Only address low-scoring categories.\n"
        "- Use both judge verdicts and ground-truth-backed case context.\n"
        "- Keep final prompt concise and production-safe."
        f"{growth_hint}\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"Low-scoring categories:\n{low_scoring_categories}\n\n"
        f"Judge verdict context:\n{verdict_context}\n\n"
        f"Ground truth:\n{ground_truth or 'N/A'}\n\n"
        f"Actions:\n{action_lines}\n"
    )


def build_safe_runtime_instruction(
    *,
    current_prompt: str,
    low_scoring_categories: str,
    action_lines: str,
) -> str:
    """Render compact runtime instruction for filter-retry path."""
    return (
        "You are Refiner Agent.\n"
        "Return strict JSON:\n"
        "{\n"
        '  "refined_prompt": "string",\n'
        '  "summary": "string"\n'
        "}\n"
        "Rules:\n"
        "- Keep wording neutral and operational.\n"
        "- Apply at most 2 high-priority fixes.\n"
        "- Keep output concise and production-safe.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"Low-scoring categories:\n{low_scoring_categories}\n\n"
        f"Fixes:\n{action_lines}\n"
    )
