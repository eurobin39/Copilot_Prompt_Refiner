from __future__ import annotations

from typing import Any

from copilot_prompt_refiner.agents.base import TextGenerationRuntime
from copilot_prompt_refiner.agents.refine.definition import (
    OUTPUT_FORMAT_PATCH,
    RUNTIME_SYSTEM_PROMPT,
    SAFETY_POLICY_PATCH,
    SAFE_RUNTIME_SYSTEM_PROMPT,
    SUCCESS_CRITERIA_PATCH,
    TOOL_USAGE_PATCH,
    VALIDATION_PATCH,
    build_runtime_instruction,
    build_safe_runtime_instruction,
)
from copilot_prompt_refiner.agents.refine.tools import (
    extract_content_filter_diagnostics,
    extract_diagnostics_from_payload,
    extract_json_object,
    is_content_filter_error,
    sanitize_for_policy,
)
from copilot_prompt_refiner.models import AgentCase, JudgeResult, PrioritizedAction, PromptRevision


class RefineAgent:
    """Apply Judge outputs to produce controlled prompt revisions.

    The agent prefers runtime generation for nuanced edits but always keeps
    deterministic fallback paths so refinement can continue under failures.
    """

    def __init__(
        self,
        runtime: TextGenerationRuntime | None = None,
        strict_runtime: bool = True,
        max_prompt_growth_ratio: float | None = None,
        max_actions: int = 3,
    ) -> None:
        """Configure refinement behavior and optional runtime dependencies.

        `strict_runtime` controls whether runtime failures abort refinement or
        fall back to deterministic patching logic.
        """
        self.runtime = runtime
        self.strict_runtime = strict_runtime
        self.max_prompt_growth_ratio = max_prompt_growth_ratio
        self.max_actions = max_actions

    def refine(self, case: AgentCase, judge_result: JudgeResult) -> PromptRevision:
        """Produce one prompt revision from case context and Judge findings.

        Runtime-based refinement is attempted first; on runtime/filter failures,
        the method falls back to heuristic patching while recording diagnostics.
        """
        selected_actions = self._select_actions(judge_result)
        targeted_failures = [item.linked_failure_type for item in selected_actions]
        action_texts = [item.action for item in selected_actions]
        fallback_note: str | None = None

        if self.runtime:
            try:
                refined_prompt, runtime_notes = self._refine_with_runtime(
                    case=case,
                    judge_result=judge_result,
                    selected_actions=selected_actions,
                )
                refined_prompt, ratio, trim_note = self._limit_growth(
                    original_prompt=case.system_prompt,
                    refined_prompt=refined_prompt,
                    selected_actions=selected_actions,
                )
                notes = runtime_notes
                if trim_note:
                    notes.append(trim_note)
                return PromptRevision(
                    original_prompt=case.system_prompt,
                    refined_prompt=refined_prompt,
                    change_log=notes,
                    patch_notes=notes,
                    applied_actions=action_texts,
                    targeted_failures=targeted_failures,
                    prompt_growth_ratio=ratio,
                    confidence=0.82,
                )
            except Exception as exc:
                if self.strict_runtime and not self._is_content_filter_error(exc):
                    raise
                if self._is_content_filter_error(exc):
                    details = self._extract_content_filter_diagnostics(exc)
                    fallback_note = (
                        "Runtime refine call was blocked by Azure content filter; "
                        f"used heuristic fallback patch. {details}"
                    )
                else:
                    fallback_note = "Runtime refine call failed; used heuristic fallback patch."

        revision = self._refine_heuristically(case, judge_result, selected_actions)
        if fallback_note:
            revision.change_log.insert(0, fallback_note)
            revision.patch_notes.insert(0, fallback_note)
        return revision

    def _refine_with_runtime(
        self,
        *,
        case: AgentCase,
        judge_result: JudgeResult,
        selected_actions: list[PrioritizedAction],
    ) -> tuple[str, list[str]]:
        """Run LLM refinement with compact, policy-safe Judge context.

        The runtime is asked for strict JSON so refined prompt text and summary
        can be parsed deterministically and attached to change notes.
        """
        action_lines = self._compact_action_lines(selected_actions)
        low_scoring_categories = self._low_scoring_categories(judge_result)
        verdict_context = self._verdict_context(judge_result)
        growth_hint = (
            f"\n- Keep prompt growth <= {int(self.max_prompt_growth_ratio * 100)}%."
            if self.max_prompt_growth_ratio is not None and self.max_prompt_growth_ratio > 0
            else ""
        )
        user_instruction = build_runtime_instruction(
            current_prompt=case.system_prompt,
            low_scoring_categories=low_scoring_categories,
            verdict_context=verdict_context,
            ground_truth=case.ground_truth,
            action_lines=action_lines,
            growth_hint=growth_hint,
        )
        try:
            raw = self.runtime.complete(
                system_prompt=RUNTIME_SYSTEM_PROMPT,
                user_input=user_instruction,
            )
            refined, summary = self._parse_refine_response(raw)
            notes = [
                "Refined via Microsoft Agent Framework runtime.",
                "Applied Judge prioritized actions with compact runtime instruction.",
            ]
            if summary:
                notes.append(f"Runtime summary: {summary}")
            return refined.strip(), notes
        except Exception as exc:
            if not self._is_content_filter_error(exc):
                raise
            details = self._extract_content_filter_diagnostics(exc)

            safe_instruction = build_safe_runtime_instruction(
                current_prompt=case.system_prompt,
                low_scoring_categories=low_scoring_categories,
                action_lines=action_lines,
            )
            raw = self.runtime.complete(
                system_prompt=SAFE_RUNTIME_SYSTEM_PROMPT,
                user_input=safe_instruction,
            )
            refined, summary = self._parse_refine_response(raw)
            notes = [
                f"Primary runtime refine prompt was filtered; retried with ultra-compact context. {details}",
                "Refined via Microsoft Agent Framework runtime.",
            ]
            if summary:
                notes.append(f"Runtime summary: {summary}")
            return refined.strip(), notes

    def _refine_heuristically(
        self,
        case: AgentCase,
        judge_result: JudgeResult,
        selected_actions: list[PrioritizedAction],
    ) -> PromptRevision:
        """Apply deterministic section-level patches without any runtime dependency.

        This fallback targets only high-priority failures and keeps behavior
        predictable in offline, strict-security, or content-filtered scenarios.
        """
        prompt = case.system_prompt.strip()
        lower_prompt = prompt.lower()
        patch_notes: list[str] = []

        if not selected_actions:
            selected_actions = self._select_actions(judge_result)

        for action in selected_actions:
            section = action.target_section.lower()
            if "output format" in section or "json" in action.action.lower():
                if "output format" not in lower_prompt:
                    prompt += OUTPUT_FORMAT_PATCH
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Output Format patch for {action.linked_failure_type}."
                    )
            elif "tool" in section:
                if "tool usage rules" not in lower_prompt and "tool policy" not in lower_prompt:
                    prompt += TOOL_USAGE_PATCH
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Tool Usage Rules patch for {action.linked_failure_type}."
                    )
            elif "safety" in section or "policy" in section:
                if "safety/policy" not in lower_prompt and "safety" not in lower_prompt:
                    prompt += SAFETY_POLICY_PATCH
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Safety/Policy patch for {action.linked_failure_type}."
                    )
            elif "success criteria" in section:
                if "success criteria" not in lower_prompt:
                    prompt += SUCCESS_CRITERIA_PATCH
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Success Criteria patch for {action.linked_failure_type}."
                    )
            else:
                if "validation" not in lower_prompt:
                    prompt += VALIDATION_PATCH
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Validation patch for {action.linked_failure_type}."
                    )

        if not patch_notes:
            prompt += VALIDATION_PATCH
            patch_notes.append("Added minimal validation patch.")

        prompt, ratio, trim_note = self._limit_growth(
            original_prompt=case.system_prompt,
            refined_prompt=prompt,
            selected_actions=selected_actions,
        )
        if trim_note:
            patch_notes.append(trim_note)

        return PromptRevision(
            original_prompt=case.system_prompt,
            refined_prompt=prompt,
            change_log=list(patch_notes),
            patch_notes=list(patch_notes),
            applied_actions=[item.action for item in selected_actions],
            targeted_failures=[item.linked_failure_type for item in selected_actions],
            prompt_growth_ratio=ratio,
            confidence=0.74,
        )

    def _select_actions(self, judge_result: JudgeResult) -> list[PrioritizedAction]:
        """Select a bounded action list for the current refinement step.

        Structured `prioritized_actions` are preferred; plain string
        recommendations are wrapped into generic actions as compatibility fallback.
        """
        if judge_result.prioritized_actions:
            return judge_result.prioritized_actions[: self.max_actions]

        fallback: list[PrioritizedAction] = []
        for index, action in enumerate(judge_result.recommended_actions[: self.max_actions]):
            fallback.append(
                PrioritizedAction(
                    priority=index + 1,
                    action=action,
                    target_section="General Constraints",
                    rationale="Derived from legacy recommended_actions.",
                    linked_failure_type="GENERAL_QUALITY",
                )
            )
        return fallback

    def _limit_growth(
        self,
        *,
        original_prompt: str,
        refined_prompt: str,
        selected_actions: list[PrioritizedAction],
    ) -> tuple[str, float, str | None]:
        """Enforce optional prompt-growth budget and trim patches when needed.

        If generated edits exceed the configured budget, only a compact minimal
        patch is retained to avoid broad rewrites and token-cost explosions.
        """
        original_len = max(len(original_prompt.strip()), 1)
        refined_len = len(refined_prompt.strip())
        growth_ratio = (refined_len - original_len) / float(original_len)
        if self.max_prompt_growth_ratio is None or self.max_prompt_growth_ratio <= 0:
            return refined_prompt, max(0.0, growth_ratio), None

        if growth_ratio <= self.max_prompt_growth_ratio:
            return refined_prompt, max(0.0, growth_ratio), None

        # Keep only the highest-priority single patch to avoid broad rewrites.
        compact = original_prompt.strip()
        if selected_actions:
            top_action = selected_actions[0]
            minimal_block = "## Minimal Patch\n"
            # Force upper bound by truncating action text to fit growth budget.
            max_len = int(original_len * (1.0 + self.max_prompt_growth_ratio))
            prefix = compact
            if "## Minimal Patch" not in compact:
                prefix = f"{compact}\n\n{minimal_block}- "
            else:
                prefix = f"{compact}\n- "

            available = max_len - len(prefix)
            if available <= 0:
                return (
                    original_prompt.strip(),
                    0.0,
                    "Skipped patch because growth budget is too small for minimal section markup.",
                )
            action_text = top_action.action.strip()
            if available > 3 and len(action_text) > available:
                action_text = action_text[: max(0, available - 3)].rstrip() + "..."

            if "## Minimal Patch" in compact:
                compact = f"{compact}\n- {action_text}".rstrip()
            else:
                compact = f"{compact}\n\n{minimal_block}- {action_text}".rstrip()

        compact_len = len(compact.strip())
        compact_ratio = (compact_len - original_len) / float(original_len)
        return (
            compact,
            max(0.0, compact_ratio),
            "Trimmed patch to keep prompt-growth budget within configured limit.",
        )

    def _is_content_filter_error(self, exc: Exception) -> bool:
        """Detect Azure/OpenAI content-filter style failures from rich or plain errors.

        Supports both structured `AzureOpenAIRequestError` payloads and generic
        exception text markers for robust fallback triggering.
        """
        return is_content_filter_error(exc)

    def _extract_content_filter_diagnostics(self, exc: Exception) -> str:
        """Extract human-readable content filter diagnostics from caught exceptions.

        Diagnostics are appended to patch notes so operators can trace why
        runtime generation was skipped or retried with safer instructions.
        """
        return extract_content_filter_diagnostics(exc)

    def _extract_diagnostics_from_payload(
        self,
        payload: dict[str, Any],
        route: str | None = None,
        status_code: int | None = None,
    ) -> str:
        """Format structured Azure error payload into concise diagnostic text.

        The formatter reports route/status/codes and any triggered categories to
        make filter outcomes observable without dumping full provider payloads.
        """
        return extract_diagnostics_from_payload(
            payload=payload,
            route=route,
            status_code=status_code,
        )

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        """Recover a JSON object from raw text when strict parsing is not guaranteed.

        Returns `None` if no object can be decoded, allowing callers to keep
        graceful fallback behavior instead of raising parse exceptions.
        """
        return extract_json_object(text)

    def _sanitize_for_policy(self, text: str, max_len: int = 260) -> str:
        """Redact high-risk jailbreak phrases and enforce compact text length.

        This keeps runtime prompts operational while reducing policy-triggering
        wording in model-facing action and verdict context snippets.
        """
        return sanitize_for_policy(text=text, max_len=max_len)

    def _compact_action_lines(self, selected_actions: list[PrioritizedAction]) -> str:
        """Build short bullet lines from selected actions for runtime prompts.

        Limiting action text length and count helps maintain prompt focus and
        reduces the chance of content-filter hits during refinement calls.
        """
        if not selected_actions:
            return "- Improve format consistency."
        lines = []
        for item in selected_actions[:3]:
            text = self._sanitize_for_policy(item.action, max_len=140)
            lines.append(f"- {text}")
        return "\n".join(lines)

    def _low_scoring_categories(self, judge_result: JudgeResult) -> str:
        """Render low-scoring evaluator metrics as concise bullet text.

        Runtime refinement uses this signal to constrain edits to weak areas
        instead of rewriting already-strong sections.
        """
        categories: list[str] = []
        for score in sorted(judge_result.evaluator_scores, key=lambda item: item.score):
            if score.score < 0.7:
                categories.append(f"- {score.metric}: {score.score:.2f}")
        if not categories:
            return "- none"
        return "\n".join(categories[:5])

    def _verdict_context(self, judge_result: JudgeResult) -> str:
        """Render top per-model verdicts into a compact runtime context block.

        Review reasoning is sanitized and truncated to preserve signal while
        minimizing policy-sensitive or overly verbose instructions.
        """
        lines: list[str] = []
        for review in judge_result.per_model_reviews[:3]:
            tags = ",".join(review.failure_tags[:3]) if review.failure_tags else "none"
            lines.append(
                f"- {review.reviewer_model}: score={review.score_0_to_10:.1f}/10, tags={tags}, "
                f"reason={self._sanitize_for_policy(review.reasoning, max_len=120)}"
            )
        if not lines:
            return "- no per-model verdicts"
        return "\n".join(lines)

    def _parse_refine_response(self, raw: str) -> tuple[str, str]:
        """Parse runtime output into `(refined_prompt, summary)` with safe fallback.

        If strict JSON parsing fails, raw text is treated as prompt output so the
        refinement pipeline can still return a usable revision.
        """
        parsed = self._extract_json_object(raw)
        if parsed is None:
            return raw.strip(), ""

        refined = parsed.get("refined_prompt")
        summary = parsed.get("summary")
        if not isinstance(refined, str) or not refined.strip():
            return raw.strip(), ""
        summary_text = summary.strip() if isinstance(summary, str) else ""
        return refined.strip(), summary_text
