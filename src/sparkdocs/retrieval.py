"""Small deterministic evidence selector for the Wave 2 foundation."""

import re

from sparkdocs.models import IngestedDocument, Section

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}")
STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "is",
    "it",
    "of",
    "does",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS}


def select_evidence(
    document: IngestedDocument,
    question: str,
    *,
    limit: int = 6,
    max_characters: int = 24_000,
) -> tuple[Section, ...]:
    """Return count- and character-bounded matches without an all-document fallback."""

    if limit < 1 or max_characters < 1:
        raise ValueError("Evidence limits must be positive.")
    question_tokens = _tokens(question)
    if not question_tokens:
        return ()
    scored: list[tuple[int, int, Section]] = []
    for index, section in enumerate(document.sections):
        score = len(question_tokens & _tokens(section.text))
        if score:
            scored.append((score, -index, section))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected: list[Section] = []
    remaining = max_characters
    for _, _, section in scored[:limit]:
        if remaining <= 0:
            break
        text = section.text[:remaining]
        if text:
            selected.append(Section(locator=section.locator, text=text))
            remaining -= len(text)
    return tuple(selected)
