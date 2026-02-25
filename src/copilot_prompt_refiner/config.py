from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or default


def _as_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def load_dotenv(dotenv_path: str | Path = ".env", override: bool = False) -> None:
    """Minimal .env loader without external dependencies."""
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class RuntimeConfig:
    default_pass_threshold: float = 0.72
    default_workspace: str = "."
    default_max_iters: int = 3
    default_model: str = "gpt-4.1-mini"
    use_microsoft_agent_framework: bool = True
    strict_microsoft_agent_framework: bool = True
    judge_review_models: list[str] = field(
        default_factory=lambda: ["gpt-4.1-mini", "mistral-large", "grok-2-latest"]
    )
    judge_disagreement_threshold: float = 2.5
    refine_max_prompt_growth_ratio: float | None = None
    refine_max_actions: int = 3
    maf_endpoint: str | None = None
    maf_api_key: str | None = None
    maf_api_version: str | None = None

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            default_pass_threshold=float(
                os.getenv("PROMPT_REFINER_PASS_THRESHOLD", "0.72")
            ),
            default_workspace=os.getenv("PROMPT_REFINER_WORKSPACE", "."),
            default_max_iters=int(os.getenv("PROMPT_REFINER_MAX_ITERS", "3")),
            default_model=os.getenv(
                "AZURE_OPENAI_MODEL",
                os.getenv("MAF_MODEL", os.getenv("PROMPT_REFINER_MODEL", "gpt-4.1-mini")),
            ),
            use_microsoft_agent_framework=_as_bool(
                os.getenv("PROMPT_REFINER_USE_MAF"), True
            ),
            strict_microsoft_agent_framework=_as_bool(
                os.getenv("PROMPT_REFINER_STRICT_MAF"), True
            ),
            judge_review_models=_as_csv(
                os.getenv("PROMPT_REFINER_JUDGE_MODELS"),
                ["gpt-4.1-mini", "mistral-large", "grok-2-latest"],
            ),
            judge_disagreement_threshold=float(
                os.getenv("PROMPT_REFINER_JUDGE_DISAGREEMENT_THRESHOLD", "2.5")
            ),
            refine_max_prompt_growth_ratio=_as_optional_float(
                os.getenv("PROMPT_REFINER_REFINE_MAX_GROWTH")
            ),
            refine_max_actions=int(os.getenv("PROMPT_REFINER_REFINE_MAX_ACTIONS", "3")),
            maf_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", os.getenv("MAF_ENDPOINT")),
            maf_api_key=os.getenv("AZURE_OPENAI_API_KEY", os.getenv("MAF_API_KEY")),
            maf_api_version=os.getenv(
                "OPENAI_API_VERSION",
                os.getenv("AZURE_OPENAI_API_VERSION", os.getenv("MAF_API_VERSION")),
            ),
        )
