from types import SimpleNamespace
from typing import Any

import pytest

from sparkdocs.errors import ProviderResponseLimitError
from sparkdocs.providers import GeminiProvider


class FakeChatModel:
    def stream(self, messages: list[Any]) -> list[SimpleNamespace]:
        assert messages[0].content == "hello"
        return [SimpleNamespace(content="safe "), SimpleNamespace(content="response")]


def test_provider_passes_key_directly_without_environment_mutation(monkeypatch: Any) -> None:
    captured: dict[str, str] = {}

    def factory(**kwargs: str) -> FakeChatModel:
        captured.update(kwargs)
        return FakeChatModel()

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider("session-key", client_factory=factory)

    assert provider.generate("hello") == "safe response"
    assert captured["google_api_key"] == "session-key"
    assert "GOOGLE_API_KEY" not in __import__("os").environ


def test_provider_bounds_streamed_response_size() -> None:
    provider = GeminiProvider(
        "session-key",
        max_response_characters=5,
        client_factory=lambda **_: FakeChatModel(),
    )

    with pytest.raises(ProviderResponseLimitError, match="safety limit"):
        provider.generate("hello")
