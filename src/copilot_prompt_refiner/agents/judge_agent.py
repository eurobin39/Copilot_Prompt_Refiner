"""Backward-compatible import shim for JudgeAgent.

New module path:
- copilot_prompt_refiner.agents.judge.agent
"""

from copilot_prompt_refiner.agents.judge.agent import JudgeAgent

__all__ = ["JudgeAgent"]
