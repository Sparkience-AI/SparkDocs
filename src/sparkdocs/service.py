"""Framework-independent SparkDocs application service."""

import re
from dataclasses import dataclass

from sparkdocs.errors import QuestionError
from sparkdocs.models import DocumentIndex
from sparkdocs.prompts import build_grounded_answer_prompt
from sparkdocs.providers import TextProvider
from sparkdocs.retrieval import HybridRetriever

CITATION_PATTERN = re.compile(r"\[([^\[\]]{1,200})\]")
ABSTENTION_MARKERS = ("does not provide enough information", "insufficient evidence")


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    evidence_locators: tuple[str, ...]
    cited_locators: tuple[str, ...]


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

    def answer(self, index: DocumentIndex, question: str) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise QuestionError("Enter a question before asking SparkDocs.")
        if len(normalized_question) > self._max_question_characters:
            raise QuestionError("The question exceeds the configured character limit.")
        evidence = self._retriever.retrieve(
            index,
            normalized_question,
            max_characters=self._max_evidence_characters,
        )
        if not evidence:
            raise QuestionError("No matching evidence was found in the document.")
        prompt = build_grounded_answer_prompt(normalized_question, evidence)
        answer = self._provider.generate(prompt)
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
        )
