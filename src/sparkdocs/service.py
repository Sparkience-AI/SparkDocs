"""Framework-independent SparkDocs application service."""

from dataclasses import dataclass

from sparkdocs.errors import QuestionError
from sparkdocs.models import IngestedDocument
from sparkdocs.prompts import build_grounded_answer_prompt
from sparkdocs.providers import TextProvider
from sparkdocs.retrieval import select_evidence


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    evidence_locators: tuple[str, ...]


class DocumentQuestionService:
    """Bounded question-answering facade over the provisional lexical selector."""

    def __init__(
        self,
        provider: TextProvider,
        *,
        max_question_characters: int = 2_000,
        max_evidence_characters: int = 24_000,
    ) -> None:
        self._provider = provider
        self._max_question_characters = max_question_characters
        self._max_evidence_characters = max_evidence_characters

    def answer(self, document: IngestedDocument, question: str) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise QuestionError("Enter a question before asking SparkDocs.")
        if len(normalized_question) > self._max_question_characters:
            raise QuestionError("The question exceeds the configured character limit.")
        evidence = select_evidence(
            document,
            normalized_question,
            max_characters=self._max_evidence_characters,
        )
        if not evidence:
            raise QuestionError("No matching evidence was found in the document.")
        prompt = build_grounded_answer_prompt(normalized_question, evidence)
        answer = self._provider.generate(prompt)
        return AnswerResult(
            answer=answer,
            evidence_locators=tuple(section.locator for section in evidence),
        )
