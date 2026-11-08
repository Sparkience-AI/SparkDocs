"""Session-owned model provider adapters with direct BYOK configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from sparkdocs.errors import ProviderResponseLimitError


class TextProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from one application-owned prompt."""


class ProviderKind(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


PROVIDER_LABELS = {
    ProviderKind.GEMINI: "Google Gemini",
    ProviderKind.OPENAI: "OpenAI",
    ProviderKind.ANTHROPIC: "Anthropic Claude",
}

DEFAULT_MODELS = {
    ProviderKind.GEMINI: "gemini-3.8-flash",
    ProviderKind.OPENAI: "gpt-6-luna",
    ProviderKind.ANTHROPIC: "claude-haiku-5-5",
}


class StreamingChatModel(Protocol):
    def stream(self, messages: list[HumanMessage]) -> Iterable[Any]: ...


ChatModelFactory = Callable[..., StreamingChatModel]
ClientFactory = Callable[..., Any]


def _validate_configuration(
    api_key: str, model: str, max_response_characters: int, timeout_seconds: float
) -> tuple[str, str]:
    normalized_key = api_key.strip()
    normalized_model = model.strip()
    if not normalized_key:
        raise ValueError("An API key is required for the selected provider.")
    if not normalized_model:
        raise ValueError("A model identifier is required.")
    if max_response_characters < 1:
        raise ValueError("The response character limit must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("The provider timeout must be positive.")
    return normalized_key, normalized_model


def _enforce_response_limit(text: str, limit: int) -> str:
    if len(text) > limit:
        raise ProviderResponseLimitError("The model response exceeded the safety limit.")
    return text


class GeminiProvider:
    """Gemini client whose key remains scoped to the active app session."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODELS[ProviderKind.GEMINI],
        max_response_characters: int = 32_000,
        timeout_seconds: float = 60.0,
        client_factory: ChatModelFactory | None = None,
    ) -> None:
        api_key, model = _validate_configuration(
            api_key, model, max_response_characters, timeout_seconds
        )
        factory = client_factory or cast(ChatModelFactory, ChatGoogleGenerativeAI)
        self._client = factory(model=model, google_api_key=api_key, timeout=timeout_seconds)
        self._max_response_characters = max_response_characters

    def generate(self, prompt: str) -> str:
        parts: list[str] = []
        characters = 0
        for chunk in self._client.stream([HumanMessage(content=prompt)]):
            content = str(chunk.content)
            characters += len(content)
            if characters > self._max_response_characters:
                raise ProviderResponseLimitError("The model response exceeded the safety limit.")
            parts.append(content)
        return "".join(parts)


class OpenAIProvider:
    """OpenAI Responses API adapter with storage disabled and a session-owned key."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODELS[ProviderKind.OPENAI],
        max_response_characters: int = 32_000,
        max_output_tokens: int = 8_192,
        timeout_seconds: float = 60.0,
        client_factory: ClientFactory | None = None,
    ) -> None:
        api_key, model = _validate_configuration(
            api_key, model, max_response_characters, timeout_seconds
        )
        if max_output_tokens < 1:
            raise ValueError("The provider output-token limit must be positive.")
        if client_factory is None:
            from openai import OpenAI

            client_factory = OpenAI
        self._client = client_factory(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._max_response_characters = max_response_characters

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        return _enforce_response_limit(str(response.output_text), self._max_response_characters)


class AnthropicProvider:
    """Anthropic Messages API adapter with a session-owned key."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODELS[ProviderKind.ANTHROPIC],
        max_response_characters: int = 32_000,
        max_output_tokens: int = 8_192,
        timeout_seconds: float = 60.0,
        client_factory: ClientFactory | None = None,
    ) -> None:
        api_key, model = _validate_configuration(
            api_key, model, max_response_characters, timeout_seconds
        )
        if max_output_tokens < 1:
            raise ValueError("The provider output-token limit must be positive.")
        if client_factory is None:
            from anthropic import Anthropic

            client_factory = Anthropic
        self._client = client_factory(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._max_response_characters = max_response_characters

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            str(block.text) for block in response.content if getattr(block, "type", None) == "text"
        )
        return _enforce_response_limit(text, self._max_response_characters)


def create_provider(kind: ProviderKind, api_key: str, model: str) -> TextProvider:
    """Construct the explicitly selected provider without mutating process-global state."""

    if kind is ProviderKind.GEMINI:
        return GeminiProvider(api_key, model=model)
    if kind is ProviderKind.OPENAI:
        return OpenAIProvider(api_key, model=model)
    if kind is ProviderKind.ANTHROPIC:
        return AnthropicProvider(api_key, model=model)
    raise ValueError("Unsupported model provider.")
