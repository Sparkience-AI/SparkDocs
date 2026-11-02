"""Session-owned model provider adapters."""

from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from sparkdocs.errors import ProviderResponseLimitError


class TextProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from one application-owned prompt."""


class StreamingChatModel(Protocol):
    def stream(self, messages: list[HumanMessage]) -> Iterable[Any]: ...


ChatModelFactory = Callable[..., StreamingChatModel]


class GeminiProvider:
    """Gemini client whose key remains scoped to the active app session."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "models/gemini-2.0-flash",
        max_response_characters: int = 32_000,
        client_factory: ChatModelFactory | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A Google API key is required.")
        if max_response_characters < 1:
            raise ValueError("The response character limit must be positive.")
        factory = client_factory or cast(ChatModelFactory, ChatGoogleGenerativeAI)
        self._client = factory(model=model, google_api_key=api_key)
        self._max_response_characters = max_response_characters

    def generate(self, prompt: str) -> str:
        chunks = self._client.stream([HumanMessage(content=prompt)])
        parts: list[str] = []
        characters = 0
        for chunk in chunks:
            content = str(chunk.content)
            characters += len(content)
            if characters > self._max_response_characters:
                raise ProviderResponseLimitError("The model response exceeded the safety limit.")
            parts.append(content)
        return "".join(parts)
