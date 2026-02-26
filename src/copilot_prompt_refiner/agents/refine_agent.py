from __future__ import annotations

import json
import re
from typing import Any

from copilot_prompt_refiner.agents.base import TextGenerationRuntime
from copilot_prompt_refiner.agents.microsoft_agent_framework import AzureOpenAIRequestError
from copilot_prompt_refiner.models import AgentCase, JudgeResult, PrioritizedAction, PromptRevision


class RefineAgent:
    def __init__(
        self,
        runtime: TextGenerationRuntime | None = None,
        strict_runtime: bool = True,
        max_prompt_growth_ratio: float | None = None,
        max_actions: int = 3,
    ) -> None:
        self.runtime = runtime
        self.strict_runtime = strict_runtime
        self.max_prompt_growth_ratio = max_prompt_growth_ratio
        self.max_actions = max_actions

    def refine(self, case: AgentCase, judge_result: JudgeResult) -> PromptRevision:
        """Apply Judge findings to produce a revised prompt with traceable patch notes."""
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
        """Call LLM runtime with compact Judge context and parse strict JSON output."""
        action_lines = self._compact_action_lines(selected_actions)
        low_scoring_categories = self._low_scoring_categories(judge_result)
        verdict_context = self._verdict_context(judge_result)
        growth_hint = (
            f"\n- Keep prompt growth <= {int(self.max_prompt_growth_ratio * 100)}%."
            if self.max_prompt_growth_ratio is not None and self.max_prompt_growth_ratio > 0
            else ""
        )
        user_instruction = (
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
            f"Current prompt:\n{case.system_prompt}\n\n"
            f"Low-scoring categories:\n{low_scoring_categories}\n\n"
            f"Judge verdict context:\n{verdict_context}\n\n"
            f"Ground truth:\n{case.ground_truth or 'N/A'}\n\n"
            f"Actions:\n{action_lines}\n"
        )
        try:
            raw = self.runtime.complete(
                system_prompt=(
                    "You are Refiner Agent for production prompts."
                ),
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

            safe_instruction = (
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
                f"Current prompt:\n{case.system_prompt}\n\n"
                f"Low-scoring categories:\n{low_scoring_categories}\n\n"
                f"Fixes:\n{action_lines}\n"
            )
            raw = self.runtime.complete(
                system_prompt=(
                    "You are a concise prompt editor."
                ),
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
        """Deterministic fallback patcher used when runtime is unavailable or filtered."""
        prompt = case.system_prompt.strip()
        lower_prompt = prompt.lower()
        patch_notes: list[str] = []

        if not selected_actions:
            selected_actions = self._select_actions(judge_result)

        for action in selected_actions:
            section = action.target_section.lower()
            if "output format" in section or "json" in action.action.lower():
                if "output format" not in lower_prompt:
                    prompt += (
                        "\n\n## Output Format\n"
                        "- Always return valid JSON.\n"
                        "- Include keys: decision, evidence, assumptions, next_actions.\n"
                        "- If data is missing, set the field to null and explain why."
                    )
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Output Format patch for {action.linked_failure_type}."
                    )
            elif "tool" in section:
                if "tool usage rules" not in lower_prompt and "tool policy" not in lower_prompt:
                    prompt += (
                        "\n\n## Tool Usage Rules\n"
                        "- Use tools when factual verification is required.\n"
                        "- Cite which tool output supports each critical claim.\n"
                        "- If tool access fails, report limitation and fallback."
                    )
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Tool Usage Rules patch for {action.linked_failure_type}."
                    )
            elif "safety" in section or "policy" in section:
                if "safety/policy" not in lower_prompt and "safety" not in lower_prompt:
                    prompt += (
                        "\n\n## Safety/Policy\n"
                        "- Refuse requests that violate policy constraints.\n"
                        "- Do not output restricted or unsafe instructions.\n"
                        "- When refusing, provide a brief safe alternative."
                    )
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Safety/Policy patch for {action.linked_failure_type}."
                    )
            elif "success criteria" in section:
                if "success criteria" not in lower_prompt:
                    prompt += (
                        "\n\n## Success Criteria\n"
                        "- Align output with provided ground truth and constraints.\n"
                        "- Avoid unsupported claims; tie conclusions to evidence."
                    )
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Success Criteria patch for {action.linked_failure_type}."
                    )
            else:
                if "validation" not in lower_prompt:
                    prompt += (
                        "\n\n## Validation\n"
                        "- Before final answer, verify constraints, tool evidence, and format."
                    )
                    lower_prompt = prompt.lower()
                    patch_notes.append(
                        f"Added Validation patch for {action.linked_failure_type}."
                    )

        if not patch_notes:
            prompt += (
                "\n\n## Validation\n"
                "- Before final answer, verify constraints, tool evidence, and format."
            )
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
        """Prefer prioritized actions from Judge; fallback to legacy recommendations."""
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
        if isinstance(exc, AzureOpenAIRequestError):
            payload = exc.response_json.get("error")
            if isinstance(payload, dict):
                code = str(payload.get("code", "")).lower()
                if "content_filter" in code:
                    return True
                inner = payload.get("innererror")
                if isinstance(inner, dict):
                    inner_code = str(inner.get("code", "")).lower()
                    if "responsibleai" in inner_code:
                        return True

        text = str(exc).lower()
        markers = (
            "content_filter",
            "responsibleaipolicyviolation",
            "jailbreak",
            "filtered",
        )
        return any(marker in text for marker in markers)

    def _extract_content_filter_diagnostics(self, exc: Exception) -> str:
        if isinstance(exc, AzureOpenAIRequestError):
            return self._extract_diagnostics_from_payload(
                exc.response_json,
                route=exc.route_url,
                status_code=exc.status_code,
            )

        text = str(exc)
        maybe_json = self._extract_json_object(text)
        if maybe_json is not None:
            return self._extract_diagnostics_from_payload(maybe_json)
        return "No structured filter details available."

    def _extract_diagnostics_from_payload(
        self,
        payload: dict[str, Any],
        route: str | None = None,
        status_code: int | None = None,
    ) -> str:
        error_obj = payload.get("error")
        if not isinstance(error_obj, dict):
            parts = []
            if status_code is not None:
                parts.append(f"status={status_code}")
            if route:
                parts.append(f"route={route}")
            return " ".join(parts) if parts else "No structured filter details available."

        code = str(error_obj.get("code", "")).strip() or "unknown"
        inner = error_obj.get("innererror")
        inner_code = ""
        categories: list[str] = []
        if isinstance(inner, dict):
            inner_code = str(inner.get("code", "")).strip()
            filter_result = inner.get("content_filter_result")
            if isinstance(filter_result, dict):
                for name, details in filter_result.items():
                    if not isinstance(details, dict):
                        continue
                    detected = bool(details.get("detected"))
                    filtered = bool(details.get("filtered"))
                    if detected or filtered:
                        categories.append(
                            f"{name}(detected={str(detected).lower()},filtered={str(filtered).lower()})"
                        )

        parts = []
        if status_code is not None:
            parts.append(f"status={status_code}")
        if route:
            parts.append(f"route={route}")
        parts.append(f"code={code}")
        if inner_code:
            parts.append(f"inner={inner_code}")
        if categories:
            parts.append("categories=" + ",".join(categories))
        return "Content filter details: " + " ".join(parts)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            maybe = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return maybe if isinstance(maybe, dict) else None

    def _sanitize_for_policy(self, text: str, max_len: int = 260) -> str:
        compact = " ".join(text.split())
        filtered = re.sub(
            r"(?i)\b(jailbreak|bypass|ignore\s+previous|override\s+policy|developer\s+mode)\b",
            "[redacted]",
            compact,
        )
        if len(filtered) <= max_len:
            return filtered
        return filtered[: max_len - 3].rstrip() + "..."

    def _compact_action_lines(self, selected_actions: list[PrioritizedAction]) -> str:
        if not selected_actions:
            return "- Improve format consistency."
        lines = []
        for item in selected_actions[:3]:
            text = self._sanitize_for_policy(item.action, max_len=140)
            lines.append(f"- {text}")
        return "\n".join(lines)

    def _low_scoring_categories(self, judge_result: JudgeResult) -> str:
        categories: list[str] = []
        for score in sorted(judge_result.evaluator_scores, key=lambda item: item.score):
            if score.score < 0.7:
                categories.append(f"- {score.metric}: {score.score:.2f}")
        if not categories:
            return "- none"
        return "\n".join(categories[:5])

    def _verdict_context(self, judge_result: JudgeResult) -> str:
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
        """Parse runtime JSON response and return refined prompt plus summary text."""
        parsed = self._extract_json_object(raw)
        if parsed is None:
            return raw.strip(), ""

        refined = parsed.get("refined_prompt")
        summary = parsed.get("summary")
        if not isinstance(refined, str) or not refined.strip():
            return raw.strip(), ""
        summary_text = summary.strip() if isinstance(summary, str) else ""
        return refined.strip(), summary_text
