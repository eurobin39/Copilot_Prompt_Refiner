from __future__ import annotations

from typing import Protocol


class TextGenerationRuntime(Protocol):
    def complete(self, *, system_prompt: str, user_input: str) -> str:
        """Return generated output for the given system/user pair."""
