"""Session-owned model provider adapters with direct BYOK configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from sparkdocs.errors import ProviderRequestError, ProviderResponseLimitError


class TextProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from one application-owned prompt."""


def explain_provider_failure(error: Exception) -> ProviderRequestError:
    """Classify OpenAI request errors without including provider-supplied data."""
    if isinstance(error, APITimeoutError):
        return ProviderRequestError("The model provider timed out. Try again.")
    if isinstance(error, APIConnectionError):
        return ProviderRequestError("Could not reach the model provider. Check your connection.")
    if isinstance(error, APIStatusError):
        if error.status_code == 401:
            return ProviderRequestError("The provider rejected the API key. Check or replace it.")
        if error.status_code == 404 or error.code == "model_not_found":
            return ProviderRequestError(
                "The model is unavailable for this account. Check the model ID and access."
            )
        if error.status_code == 403:
            return ProviderRequestError("The provider denied access. Check account permissions.")
        if error.status_code == 429:
            return ProviderRequestError("The provider rate or quota limit was reached.")
    return ProviderRequestError(
        "The model request failed. Check the key, model ID, connection, and account quota."
    )


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


def generate_text(
    provider: TextProvider, prompt: str, on_delta: Callable[[str], None] | None = None
) -> str:
    stream = getattr(type(provider), "stream_text", None)
    if on_delta is None or stream is None:
        return provider.generate(prompt)
    parts = []
    with_stream = stream(provider, prompt)
    try:
        for text in with_stream:
            parts.append(text)
            on_delta(text)
    except APIStatusError as exc:
        detail = str(exc.message).lower()
        if (
            not parts
            and exc.status_code in (400, 422)
            and "stream" in detail
            and ("not supported" in detail or "does not support" in detail)
        ):
            return provider.generate(prompt)
        raise
    finally:
        with_stream.close()
    if not parts:
        raise ProviderRequestError("The model did not return any text.")
    return "".join(parts)


def _bounded_deltas(deltas: Iterable[str], limit: int) -> Iterator[str]:
    characters = 0
    for text in deltas:
        if text:
            characters += len(text)
            if characters > limit:
                raise ProviderResponseLimitError("The model response exceeded the safety limit.")
            yield text


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
        return "".join(self.stream_text(prompt))

    def stream_text(self, prompt: str) -> Iterator[str]:
        chunks = self._client.stream([HumanMessage(content=prompt)])
        yield from _bounded_deltas(
            (chunk.content if isinstance(chunk.content, str) else chunk.text for chunk in chunks),
            self._max_response_characters,
        )


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

    def stream_text(self, prompt: str) -> Iterator[str]:
        with self._client.responses.stream(
            model=self._model, input=prompt, max_output_tokens=self._max_output_tokens, store=False
        ) as stream:
            yield from _bounded_deltas(
                (event.delta for event in stream if event.type == "response.output_text.delta"),
                self._max_response_characters,
            )
            if stream.get_final_response().status != "completed":
                raise ProviderRequestError("The model did not complete its response.")


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

    def stream_text(self, prompt: str) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from _bounded_deltas(stream.text_stream, self._max_response_characters)


def create_provider(kind: ProviderKind, api_key: str, model: str) -> TextProvider:
    """Construct the explicitly selected provider without mutating process-global state."""

    if kind is ProviderKind.GEMINI:
        return GeminiProvider(api_key, model=model)
    if kind is ProviderKind.OPENAI:
        return OpenAIProvider(api_key, model=model)
    if kind is ProviderKind.ANTHROPIC:
        return AnthropicProvider(api_key, model=model)
    raise ValueError("Unsupported model provider.")
