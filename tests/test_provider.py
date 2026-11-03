from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sparkdocs.errors import ProviderResponseLimitError
from sparkdocs.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderKind,
    create_provider,
)


class FakeGeminiModel:
    def stream(self, messages: list[Any]) -> list[SimpleNamespace]:
        assert messages[0].content == "hello"
        return [SimpleNamespace(content="safe "), SimpleNamespace(content="response")]


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="safe response")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="not returned"),
                SimpleNamespace(type="text", text="safe response"),
            ]
        )


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


def test_gemini_passes_key_directly_without_environment_mutation(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeGeminiModel:
        captured.update(kwargs)
        return FakeGeminiModel()

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider("session-key", client_factory=factory)

    assert provider.generate("hello") == "safe response"
    assert captured["google_api_key"] == "session-key"
    assert captured["model"] == "gemini-3.8-flash"
    assert "GOOGLE_API_KEY" not in __import__("os").environ


def test_openai_uses_responses_api_without_server_storage(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    client = FakeOpenAIClient()

    def factory(**kwargs: Any) -> FakeOpenAIClient:
        captured.update(kwargs)
        return client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider("session-key", client_factory=factory)

    assert provider.generate("hello") == "safe response"
    assert captured["api_key"] == "session-key"
    assert client.responses.calls == [
        {
            "model": "gpt-6-luna",
            "input": "hello",
            "max_output_tokens": 8_192,
            "store": False,
        }
    ]
    assert "OPENAI_API_KEY" not in __import__("os").environ


def test_anthropic_uses_messages_api_and_returns_only_text(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    client = FakeAnthropicClient()

    def factory(**kwargs: Any) -> FakeAnthropicClient:
        captured.update(kwargs)
        return client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider("session-key", client_factory=factory)

    assert provider.generate("hello") == "safe response"
    assert captured["api_key"] == "session-key"
    assert client.messages.calls == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 8_192,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ]
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_factory_requires_a_non_empty_key(kind: ProviderKind) -> None:
    with pytest.raises(ValueError, match="API key"):
        create_provider(kind, "", "model")


def test_all_providers_bound_response_size() -> None:
    providers = (
        GeminiProvider(
            "key", max_response_characters=5, client_factory=lambda **_: FakeGeminiModel()
        ),
        OpenAIProvider(
            "key", max_response_characters=5, client_factory=lambda **_: FakeOpenAIClient()
        ),
        AnthropicProvider(
            "key", max_response_characters=5, client_factory=lambda **_: FakeAnthropicClient()
        ),
    )

    for provider in providers:
        with pytest.raises(ProviderResponseLimitError, match="safety limit"):
            provider.generate("hello")
