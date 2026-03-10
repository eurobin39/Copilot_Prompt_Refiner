"""Backward-compatible import shim for RefineAgent.

New module path:
- copilot_prompt_refiner.agents.refine.agent
"""

from copilot_prompt_refiner.agents.refine.agent import RefineAgent

__all__ = ["RefineAgent"]
