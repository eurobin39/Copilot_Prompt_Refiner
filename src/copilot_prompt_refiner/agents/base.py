from __future__ import annotations

from typing import Protocol


class TextGenerationRuntime(Protocol):
    """Protocol for runtime adapters used by Judge and Refine agents.

    Any implementation only needs to expose a `complete` method so the rest of
    the pipeline remains provider-agnostic.
    """

    def complete(self, *, system_prompt: str, user_input: str) -> str:
        """Generate text output for a `(system_prompt, user_input)` pair.

        Callers expect plain text and handle JSON/schema parsing in their own
        layers, keeping this interface intentionally minimal.
        """
