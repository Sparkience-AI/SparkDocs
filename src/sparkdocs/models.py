"""Normalized document models with source locators."""

from dataclasses import dataclass
from enum import StrEnum


class DocumentKind(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


@dataclass(frozen=True, slots=True)
class Section:
    """A bounded unit of extracted text and its human-readable source locator."""

    locator: str
    text: str


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """Ephemeral normalized representation of one validated upload."""

    filename: str
    kind: DocumentKind
    media_type: str
    sha256: str
    sections: tuple[Section, ...]

    @property
    def character_count(self) -> int:
        return sum(len(section.text) for section in self.sections)
