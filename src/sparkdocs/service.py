"""Framework-independent SparkDocs application service."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sparkdocs.errors import InsufficientEvidenceError, ProviderResponseLimitError, QuestionError
from sparkdocs.models import DocumentIndex, Section
from sparkdocs.prompts import build_grounded_answer_prompt
from sparkdocs.providers import TextProvider, explain_provider_failure
from sparkdocs.retrieval import HybridRetriever

CITATION_PATTERN = re.compile(r"\[([^\[\]]{1,200})\]")
ABSTENTION_MARKERS = ("does not provide enough information", "insufficient evidence")
MAX_PREVIOUS_QUESTIONS = 2
MAX_CONTEXT_CHARACTERS = 600


def _bounded_questions(previous_questions: Sequence[str]) -> tuple[str, ...]:
    recent = []
    remaining = MAX_CONTEXT_CHARACTERS
    for question in reversed(previous_questions[-MAX_PREVIOUS_QUESTIONS:]):
        text = question.strip()[:remaining].rstrip()
        if text:
            recent.append(text)
            remaining -= len(text)
    return tuple(reversed(recent))


def contextualize_question(question: str, previous_questions: Sequence[str]) -> str:
    """Add bounded recent terms to the local retrieval query."""
    return "\n".join((question, *_bounded_questions(previous_questions)))


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    evidence_locators: tuple[str, ...]
    cited_locators: tuple[str, ...]
    evidence: tuple[Section, ...] = ()


class DocumentQuestionService:
    """Bounded question-answering facade over local hybrid retrieval."""

    def __init__(
        self,
        provider: TextProvider,
        retriever: HybridRetriever,
        *,
        max_question_characters: int = 2_000,
        max_evidence_characters: int = 24_000,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._max_question_characters = max_question_characters
        self._max_evidence_characters = max_evidence_characters

    def answer(
        self,
        index: DocumentIndex,
        question: str,
        *,
        previous_questions: Sequence[str] = (),
    ) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise QuestionError("Enter a question before asking SparkDocs.")
        if len(normalized_question) > self._max_question_characters:
            raise QuestionError("The question exceeds the configured character limit.")
        evidence = self._retriever.retrieve(
            index,
            contextualize_question(normalized_question, previous_questions),
            max_characters=self._max_evidence_characters,
        )
        if not evidence:
            raise InsufficientEvidenceError("No matching evidence was found in the document.")
        prompt = build_grounded_answer_prompt(
            normalized_question, evidence, previous_questions=_bounded_questions(previous_questions)
        )
        try:
            answer = self._provider.generate(prompt)
        except ProviderResponseLimitError:
            raise
        except Exception as exc:
            raise explain_provider_failure(exc) from None
        allowed = tuple(section.locator for section in evidence)
        citations = tuple(dict.fromkeys(CITATION_PATTERN.findall(answer)))
        invalid = tuple(locator for locator in citations if locator not in allowed)
        if invalid:
            raise QuestionError("The model returned a citation outside the retrieved evidence.")
        if not citations and not any(marker in answer.lower() for marker in ABSTENTION_MARKERS):
            raise QuestionError("The model answer did not include a verifiable source citation.")
        return AnswerResult(
            answer=answer,
            evidence_locators=allowed,
            cited_locators=citations,
            evidence=evidence,
        )
