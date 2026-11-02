from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf
import pytest
from docx import Document
from pptx import Presentation

from sparkdocs.errors import ExtractionLimitError, UploadRejectedError
from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.models import DocumentKind

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def make_docx(text: str) -> bytes:
    output = BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(output)
    return output.getvalue()


def make_pptx(text: str) -> bytes:
    output = BytesIO()
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    textbox = slide.shapes.add_textbox(0, 0, 1_000_000, 1_000_000)
    textbox.text = text
    presentation.save(output)
    return output.getvalue()


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


@pytest.mark.parametrize(
    ("filename", "media_type", "content", "kind", "locator"),
    [
        ("notes.pdf", "application/pdf", make_pdf("bounded PDF text"), DocumentKind.PDF, "page 1"),
        (
            "notes.docx",
            DOCX_MEDIA_TYPE,
            make_docx("bounded DOCX text"),
            DocumentKind.DOCX,
            "paragraph 1",
        ),
        (
            "notes.pptx",
            PPTX_MEDIA_TYPE,
            make_pptx("bounded PPTX text"),
            DocumentKind.PPTX,
            "slide 1",
        ),
    ],
    ids=["pdf", "docx", "pptx"],
)
def test_ingests_supported_documents_with_source_locators(
    filename: str,
    media_type: str,
    content: bytes,
    kind: DocumentKind,
    locator: str,
) -> None:
    result = ingest_upload(filename=filename, claimed_media_type=media_type, content=content)

    assert result.kind is kind
    assert locator in result.sections[0].locator
    assert "bounded" in result.sections[0].text
    assert len(result.sha256) == 64


@pytest.mark.parametrize(
    "filename",
    ["../notes.pdf", "folder/notes.pdf", "folder\\notes.pdf", "C:notes.pdf", "notes?.pdf"],
)
def test_rejects_filename_paths(filename: str) -> None:
    with pytest.raises(UploadRejectedError, match="path|characters"):
        ingest_upload(filename=filename, claimed_media_type="application/pdf", content=b"%PDF-")


def test_rejects_oversized_upload_before_parsing() -> None:
    policy = UploadPolicy(max_upload_bytes=5)

    with pytest.raises(UploadRejectedError, match="size limit"):
        ingest_upload(
            filename="notes.pdf",
            claimed_media_type="application/pdf",
            content=b"%PDF-too-large",
            policy=policy,
        )


def test_rejects_extension_media_type_mismatch() -> None:
    with pytest.raises(UploadRejectedError, match="do not match"):
        ingest_upload(
            filename="notes.pdf",
            claimed_media_type=DOCX_MEDIA_TYPE,
            content=b"%PDF-",
        )


def test_rejects_signature_mismatch() -> None:
    with pytest.raises(UploadRejectedError, match="signature"):
        ingest_upload(
            filename="notes.pdf",
            claimed_media_type="application/pdf",
            content=b"PK-not-a-pdf",
        )


def test_rejects_unsafe_path_inside_ooxml_archive() -> None:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "types")
        archive.writestr("word/document.xml", "document")
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(UploadRejectedError, match="unsafe path"):
        ingest_upload(
            filename="notes.docx",
            claimed_media_type=DOCX_MEDIA_TYPE,
            content=output.getvalue(),
        )


def test_rejects_suspicious_compression_ratio() -> None:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "types")
        archive.writestr("word/document.xml", "A" * 20_000)

    policy = UploadPolicy(max_compression_ratio=2)
    with pytest.raises(UploadRejectedError, match="compression limit"):
        ingest_upload(
            filename="notes.docx",
            claimed_media_type=DOCX_MEDIA_TYPE,
            content=output.getvalue(),
            policy=policy,
        )


def test_enforces_extracted_character_limit() -> None:
    policy = UploadPolicy(max_extracted_characters=10)

    with pytest.raises(ExtractionLimitError, match="character limit"):
        ingest_upload(
            filename="notes.docx",
            claimed_media_type=DOCX_MEDIA_TYPE,
            content=make_docx("This paragraph is longer than ten characters."),
            policy=policy,
        )
