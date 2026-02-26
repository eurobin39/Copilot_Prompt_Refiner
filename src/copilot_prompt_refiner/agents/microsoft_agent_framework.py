from __future__ import annotations

import json
import os
import socket
import ssl
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from copilot_prompt_refiner.agents.base import TextGenerationRuntime


class AzureOpenAIRequestError(RuntimeError):
    """Rich Azure OpenAI request failure with transport and payload diagnostics.

    The exception preserves HTTP status, route, and parsed JSON details so
    fallback logic can distinguish content-filter cases from infra failures.
    """

    def __init__(
        self,
        *,
        message: str,
        status_code: int | None = None,
        route_url: str | None = None,
        response_text: str = "",
        response_json: dict[str, Any] | None = None,
    ) -> None:
        """Create a request error carrying provider and transport diagnostics.

        Structured context fields allow callers to classify retryable routes,
        content-filter blocks, and endpoint misconfiguration separately.
        """
        super().__init__(message)
        self.status_code = status_code
        self.route_url = route_url
        self.response_text = response_text
        self.response_json = response_json or {}


def _as_optional_text(value: str | None) -> str | None:
    """Normalize optional text config values from constructor/env inputs.

    Empty strings are treated as `None` so configuration checks can reason
    about missing values consistently.
    """
    if value is None:
        return None
    text = value.strip()
    return text or None


def _as_bool(value: str | None, default: bool) -> bool:
    """Parse optional boolean text with fallback default behavior.

    This helper keeps env-driven flags concise and avoids repeated truthy checks
    across runtime initialization code.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class MicrosoftAgentFrameworkRuntime(TextGenerationRuntime):
    """Azure OpenAI-backed runtime used by Judge and Refine agents.

    It handles endpoint compatibility, retryable route variants, TLS setup,
    and response-shape normalization into plain completion text.
    """

    model: str = "gpt-4.1-mini"
    endpoint: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    request_timeout_sec: float = 45.0
    temperature: float | None = None
    ssl_verify: bool = True
    ca_bundle_path: str | None = None

    def __post_init__(self) -> None:
        """Resolve runtime config from explicit args and environment fallbacks.

        Supports both current `AZURE_OPENAI_*` and legacy `MAF_*` variable names
        so existing deployments continue working during migration.
        """
        self.endpoint = _as_optional_text(
            self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("MAF_ENDPOINT")
        )
        self.api_key = _as_optional_text(
            self.api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("MAF_API_KEY")
        )
        self.api_version = (
            _as_optional_text(self.api_version)
            or os.getenv("OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("MAF_API_VERSION")
            or "2024-10-21"
        )
        self.model = _as_optional_text(
            self.model
            or os.getenv("AZURE_OPENAI_MODEL")
            or os.getenv("MAF_MODEL", "gpt-4.1-mini")
        )
        timeout_value = os.getenv("PROMPT_REFINER_TIMEOUT_SEC")
        if timeout_value:
            try:
                self.request_timeout_sec = max(5.0, float(timeout_value))
            except ValueError:
                pass
        temperature_value = os.getenv("PROMPT_REFINER_TEMPERATURE")
        if temperature_value is not None and temperature_value.strip():
            try:
                self.temperature = float(temperature_value.strip())
            except ValueError:
                self.temperature = None

        self.ssl_verify = _as_bool(
            os.getenv("AZURE_OPENAI_SSL_VERIFY"),
            self.ssl_verify,
        )
        self.ca_bundle_path = (
            os.getenv("AZURE_OPENAI_CA_BUNDLE")
            or os.getenv("SSL_CERT_FILE")
            or self.ca_bundle_path
        )

    def validate_configuration(self) -> None:
        """Validate required runtime fields before making network requests.

        Early validation produces clear startup errors instead of opaque request
        failures deep inside judge/refine execution paths.
        """
        missing: list[str] = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT (or MAF_ENDPOINT)")
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY (or MAF_API_KEY)")
        if not self.model:
            missing.append("AZURE_OPENAI_MODEL (or MAF_MODEL)")

        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Azure OpenAI runtime is not fully configured. "
                f"Missing: {joined}"
            )

        assert self.endpoint is not None
        parsed_endpoint = parse.urlparse(self.endpoint)
        if parsed_endpoint.scheme not in {"https", "http"} or not parsed_endpoint.hostname:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT must be a valid absolute URL including scheme and host. "
                f"Current value: {self.endpoint!r}. "
                "Example: https://your-resource.openai.azure.com/"
            )

    def _build_chat_completions_urls(self) -> list[str]:
        """Build candidate chat-completions routes for Azure API compatibility.

        Both `v1` and versioned legacy paths are attempted so the runtime can
        handle endpoint differences across Azure resource configurations.
        """
        self.validate_configuration()
        assert self.endpoint is not None

        endpoint = self.endpoint.rstrip("/")
        urls = [f"{endpoint}/openai/v1/chat/completions"]
        if self.api_version:
            api_version_qs = parse.urlencode({"api-version": self.api_version})
            urls.append(f"{endpoint}/openai/chat/completions?{api_version_qs}")
        return urls

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Build SSL context honoring verification flag and optional CA bundle.

        This enables secure corporate/internal CA trust while still allowing
        temporary non-verified mode for diagnostics.
        """
        if not self.ssl_verify:
            return ssl._create_unverified_context()

        context = ssl.create_default_context()
        if self.ca_bundle_path:
            try:
                context.load_verify_locations(cafile=self.ca_bundle_path)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"AZURE_OPENAI_CA_BUNDLE file not found: {self.ca_bundle_path}"
                ) from exc
            except ssl.SSLError as exc:
                raise RuntimeError(
                    "Failed to load AZURE_OPENAI_CA_BUNDLE as a valid CA bundle: "
                    f"{self.ca_bundle_path}"
                ) from exc
        return context

    def complete(self, *, system_prompt: str, user_input: str) -> str:
        """Call Azure OpenAI chat completions and return normalized text content.

        The method retries compatible routes, preserves detailed failure context,
        and supports both string and segmented content response shapes.
        """
        urls = self._build_chat_completions_urls()
        ssl_context = self._build_ssl_context()
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        body = json.dumps(payload).encode("utf-8")
        raw: str | None = None
        last_http_error: error.HTTPError | None = None
        last_http_detail = ""
        last_http_route = ""
        last_url_error: error.URLError | None = None
        last_url_route = ""
        for url in urls:
            req = request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key or "",
                },
            )
            try:
                with request.urlopen(
                    req,
                    timeout=self.request_timeout_sec,
                    context=ssl_context,
                ) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break
            except error.HTTPError as exc:
                last_http_error = exc
                last_http_route = url
                detail_raw = exc.read().decode("utf-8", errors="replace")
                last_http_detail = detail_raw[:2000]
                parsed_json: dict[str, Any] | None = None
                try:
                    maybe = json.loads(detail_raw)
                    if isinstance(maybe, dict):
                        parsed_json = maybe
                except json.JSONDecodeError:
                    parsed_json = None
                # 404 in one route can still succeed with fallback route.
                if exc.code == 404:
                    continue
                raise AzureOpenAIRequestError(
                    message=f"Azure OpenAI request failed: HTTP {exc.code}. {last_http_detail}",
                    status_code=exc.code,
                    route_url=url,
                    response_text=last_http_detail,
                    response_json=parsed_json,
                ) from exc
            except error.URLError as exc:
                last_url_error = exc
                last_url_route = url
                continue

        if raw is None:
            if last_http_error is not None:
                parsed_json: dict[str, Any] | None = None
                try:
                    maybe = json.loads(last_http_detail)
                    if isinstance(maybe, dict):
                        parsed_json = maybe
                except json.JSONDecodeError:
                    parsed_json = None
                raise AzureOpenAIRequestError(
                    message=(
                        "Azure OpenAI request failed on all routes. "
                        f"Last HTTP {last_http_error.code}. {last_http_detail}"
                    ),
                    status_code=last_http_error.code,
                    route_url=last_http_route or None,
                    response_text=last_http_detail,
                    response_json=parsed_json,
                ) from last_http_error
            if last_url_error is not None:
                reason = last_url_error.reason
                route_msg = f" (route: {last_url_route})" if last_url_route else ""
                if isinstance(reason, socket.gaierror):
                    host = parse.urlparse(last_url_route).hostname if last_url_route else None
                    if host:
                        raise AzureOpenAIRequestError(
                            message=(
                                "Azure OpenAI request failed: DNS lookup failed for "
                                f"{host!r}{route_msg}. Original error: {reason}"
                            ),
                        ) from last_url_error
                raise AzureOpenAIRequestError(
                    message=f"Azure OpenAI request failed{route_msg}: {reason}",
                ) from last_url_error
            raise AzureOpenAIRequestError(
                message="Azure OpenAI request failed: no response received."
            )

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Azure OpenAI response is not valid JSON.") from exc

        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Azure OpenAI response missing choices.")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("Azure OpenAI response choice has invalid shape.")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Azure OpenAI response missing message object.")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError("Azure OpenAI response missing text content.")
