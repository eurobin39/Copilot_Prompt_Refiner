import json
import socket
import ssl
from urllib import error

import pytest

from copilot_prompt_refiner.agents.microsoft_agent_framework import (
    AzureOpenAIRequestError,
    MicrosoftAgentFrameworkRuntime,
)


class _DummyResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode("utf-8")


def test_runtime_uses_ssl_verify_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("AZURE_OPENAI_SSL_VERIFY", raising=False)

    captured_context = {}

    def _fake_urlopen(req, timeout, context):  # noqa: ANN001
        captured_context["ctx"] = context
        return _DummyResponse()

    monkeypatch.setattr(
        "copilot_prompt_refiner.agents.microsoft_agent_framework.request.urlopen",
        _fake_urlopen,
    )

    runtime = MicrosoftAgentFrameworkRuntime()
    text = runtime.complete(system_prompt="sys", user_input="user")

    assert text == "ok"
    assert isinstance(captured_context["ctx"], ssl.SSLContext)
    assert captured_context["ctx"].verify_mode == ssl.CERT_REQUIRED


def test_runtime_can_disable_ssl_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("AZURE_OPENAI_SSL_VERIFY", "false")

    captured_context = {}

    def _fake_urlopen(req, timeout, context):  # noqa: ANN001
        captured_context["ctx"] = context
        return _DummyResponse()

    monkeypatch.setattr(
        "copilot_prompt_refiner.agents.microsoft_agent_framework.request.urlopen",
        _fake_urlopen,
    )

    runtime = MicrosoftAgentFrameworkRuntime()
    text = runtime.complete(system_prompt="sys", user_input="user")

    assert text == "ok"
    assert captured_context["ctx"].verify_mode == ssl.CERT_NONE
    assert captured_context["ctx"].check_hostname is False


def test_runtime_fails_fast_for_missing_ca_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("AZURE_OPENAI_SSL_VERIFY", "true")
    monkeypatch.setenv("AZURE_OPENAI_CA_BUNDLE", "/path/does/not/exist.pem")

    runtime = MicrosoftAgentFrameworkRuntime()
    with pytest.raises(RuntimeError) as excinfo:
        runtime._build_ssl_context()

    assert "AZURE_OPENAI_CA_BUNDLE file not found" in str(excinfo.value)


def test_runtime_rejects_invalid_endpoint_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

    runtime = MicrosoftAgentFrameworkRuntime()
    with pytest.raises(RuntimeError) as excinfo:
        runtime.validate_configuration()

    assert "valid absolute URL including scheme and host" in str(excinfo.value)


def test_runtime_surfaces_dns_lookup_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

    def _fake_urlopen(req, timeout, context):  # noqa: ANN001
        raise error.URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))

    monkeypatch.setattr(
        "copilot_prompt_refiner.agents.microsoft_agent_framework.request.urlopen",
        _fake_urlopen,
    )

    runtime = MicrosoftAgentFrameworkRuntime()
    with pytest.raises(AzureOpenAIRequestError) as excinfo:
        runtime.complete(system_prompt="sys", user_input="user")

    message = str(excinfo.value)
    assert "DNS lookup failed" in message
    assert "example.openai.azure.com" in message
