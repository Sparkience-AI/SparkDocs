"""Bounded, in-memory ingestion for PDF and OOXML documents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import pymupdf
from docx import Document
from pptx import Presentation

from sparkdocs.errors import (
    EmptyDocumentError,
    ExtractionError,
    ExtractionLimitError,
    UploadRejectedError,
)
from sparkdocs.models import DocumentKind, IngestedDocument, Section


@dataclass(frozen=True, slots=True)
class FormatSpec:
    kind: DocumentKind
    extension: str
    media_type: str
    required_zip_member: str | None = None


FORMAT_SPECS = {
    ".pdf": FormatSpec(DocumentKind.PDF, ".pdf", "application/pdf"),
    ".docx": FormatSpec(
        DocumentKind.DOCX,
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "word/document.xml",
    ),
    ".pptx": FormatSpec(
        DocumentKind.PPTX,
        ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt/presentation.xml",
    ),
}


@dataclass(frozen=True, slots=True)
class UploadPolicy:
    """Resource limits enforced before and during extraction."""

    max_upload_bytes: int = 10 * 1024 * 1024
    max_uncompressed_bytes: int = 40 * 1024 * 1024
    max_zip_members: int = 2_000
    max_zip_member_bytes: int = 10 * 1024 * 1024
    max_compression_ratio: float = 100.0
    max_sections: int = 1_000
    max_extracted_characters: int = 1_000_000
    max_pdf_pages: int = 200
    max_pptx_slides: int = 200
    max_filename_characters: int = 180


class _SectionCollector:
    def __init__(self, policy: UploadPolicy) -> None:
        self._policy = policy
        self._characters = 0
        self._sections: list[Section] = []

    def add(self, locator: str, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        if len(self._sections) >= self._policy.max_sections:
            raise ExtractionLimitError("Document contains too many text sections.")
        self._characters += len(normalized)
        if self._characters > self._policy.max_extracted_characters:
            raise ExtractionLimitError("Extracted text exceeds the configured character limit.")
        self._sections.append(Section(locator=locator, text=normalized))

    def finish(self) -> tuple[Section, ...]:
        if not self._sections:
            raise EmptyDocumentError("No supported text was found in the document.")
        return tuple(self._sections)


def normalize_filename(filename: str, policy: UploadPolicy) -> str:
    """Accept one plain basename and reject paths or ambiguous names."""

    if not filename or "\x00" in filename or len(filename) > policy.max_filename_characters:
        raise UploadRejectedError("Filename is missing or exceeds the allowed length.")
    if any(ord(character) < 32 or character in '<>:"|?*' for character in filename):
        raise UploadRejectedError("Filename contains unsupported characters.")
    portable = filename.replace("\\", "/")
    if "/" in portable or portable in {".", ".."}:
        raise UploadRejectedError("Filename must not contain a path.")
    basename = PurePosixPath(portable).name
    if basename != filename or basename.startswith("."):
        raise UploadRejectedError("Filename is not a supported plain document name.")
    return basename


def _spec_for_upload(filename: str, claimed_media_type: str) -> FormatSpec:
    extension = PurePosixPath(filename).suffix.lower()
    spec = FORMAT_SPECS.get(extension)
    if spec is None:
        raise UploadRejectedError("Only PDF, DOCX, and PPTX files are supported.")
    if claimed_media_type != spec.media_type:
        raise UploadRejectedError("The filename extension and declared media type do not match.")
    return spec


def _validate_zip(content: bytes, spec: FormatSpec, policy: UploadPolicy) -> None:
    try:
        with ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > policy.max_zip_members:
                raise UploadRejectedError("The document archive contains too many members.")
            total_size = 0
            names: set[str] = set()
            for member in members:
                portable_name = member.filename.replace("\\", "/")
                path = PurePosixPath(portable_name)
                if (
                    not portable_name
                    or not path.parts
                    or path.is_absolute()
                    or ".." in path.parts
                    or ":" in path.parts[0]
                ):
                    raise UploadRejectedError("The document archive contains an unsafe path.")
                if member.flag_bits & 0x1:
                    raise UploadRejectedError("Encrypted document archives are not supported.")
                if member.file_size > policy.max_zip_member_bytes:
                    raise UploadRejectedError("A document archive member exceeds the size limit.")
                ratio = member.file_size / max(member.compress_size, 1)
                if ratio > policy.max_compression_ratio:
                    raise UploadRejectedError("The document archive exceeds the compression limit.")
                total_size += member.file_size
                if total_size > policy.max_uncompressed_bytes:
                    raise UploadRejectedError("The expanded document exceeds the size limit.")
                if portable_name in names:
                    raise UploadRejectedError("The document archive contains duplicate members.")
                names.add(portable_name)
            if "[Content_Types].xml" not in names or spec.required_zip_member not in names:
                raise UploadRejectedError("The document signature does not match its extension.")
    except BadZipFile as exc:
        raise UploadRejectedError("The document is not a valid OOXML archive.") from exc


def _validate_signature(content: bytes, spec: FormatSpec, policy: UploadPolicy) -> None:
    if spec.kind is DocumentKind.PDF:
        if not content.startswith(b"%PDF-"):
            raise UploadRejectedError("The file signature does not match a PDF document.")
        return
    if not content.startswith(b"PK"):
        raise UploadRejectedError("The file signature does not match an OOXML document.")
    _validate_zip(content, spec, policy)


def _extract_pdf(content: bytes, policy: UploadPolicy) -> tuple[Section, ...]:
    collector = _SectionCollector(policy)
    try:
        with pymupdf.open(  # type: ignore[no-untyped-call]
            stream=content, filetype="pdf"
        ) as document:
            if document.needs_pass:
                raise UploadRejectedError("Password-protected PDFs are not supported.")
            if document.page_count > policy.max_pdf_pages:
                raise ExtractionLimitError("PDF exceeds the configured page limit.")
            for index, page in enumerate(document, start=1):
                collector.add(f"PDF page {index}", page.get_text())
    except (UploadRejectedError, ExtractionLimitError):
        raise
    except Exception as exc:
        raise ExtractionError("PDF extraction failed.") from exc
    return collector.finish()


def _extract_docx(content: bytes, policy: UploadPolicy) -> tuple[Section, ...]:
    collector = _SectionCollector(policy)
    try:
        document = Document(BytesIO(content))
        for index, paragraph in enumerate(document.paragraphs, start=1):
            collector.add(f"DOCX paragraph {index}", paragraph.text)
        for table_index, table in enumerate(document.tables, start=1):
            for row_index, row in enumerate(table.rows, start=1):
                collector.add(
                    f"DOCX table {table_index}, row {row_index}",
                    " | ".join(cell.text for cell in row.cells),
                )
    except ExtractionLimitError:
        raise
    except Exception as exc:
        raise ExtractionError("DOCX extraction failed.") from exc
    return collector.finish()


def _extract_pptx(content: bytes, policy: UploadPolicy) -> tuple[Section, ...]:
    collector = _SectionCollector(policy)
    try:
        presentation = Presentation(BytesIO(content))
        if len(presentation.slides) > policy.max_pptx_slides:
            raise ExtractionLimitError("Presentation exceeds the configured slide limit.")
        for slide_index, slide in enumerate(presentation.slides, start=1):
            parts = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
            collector.add(f"PPTX slide {slide_index}", "\n".join(parts))
    except ExtractionLimitError:
        raise
    except Exception as exc:
        raise ExtractionError("PPTX extraction failed.") from exc
    return collector.finish()


def ingest_upload(
    *,
    filename: str,
    claimed_media_type: str,
    content: bytes,
    policy: UploadPolicy | None = None,
) -> IngestedDocument:
    """Validate and extract one upload entirely in memory.

    The original bytes are not written to disk. The returned document remains
    ephemeral and should be discarded when the user clears the active session.
    """

    active_policy = policy or UploadPolicy()
    safe_filename = normalize_filename(filename, active_policy)
    if not content:
        raise UploadRejectedError("The uploaded file is empty.")
    if len(content) > active_policy.max_upload_bytes:
        raise UploadRejectedError("The uploaded file exceeds the size limit.")
    spec = _spec_for_upload(safe_filename, claimed_media_type)
    _validate_signature(content, spec, active_policy)

    extractors = {
        DocumentKind.PDF: _extract_pdf,
        DocumentKind.DOCX: _extract_docx,
        DocumentKind.PPTX: _extract_pptx,
    }
    sections = extractors[spec.kind](content, active_policy)
    return IngestedDocument(
        filename=safe_filename,
        kind=spec.kind,
        media_type=spec.media_type,
        sha256=sha256(content).hexdigest(),
        sections=sections,
    )
