"""Fictional handbook for the self-hosted app walkthrough."""

from functools import lru_cache
from typing import cast

import pymupdf

SAMPLE_FILENAME = "maple-claims-handbook.pdf"
SAMPLE_QUESTIONS = (
    "How long do I have to file a battery claim for a device used outdoors?",
    "How many replacement kits can one member borrow at once?",
)

_PAGES = (
    (
        "Maple claims handbook",
        "A fictional example for SparkDocs. This handbook describes how members of "
        "the Maple Equipment Library report problems with borrowed devices.",
    ),
    (
        "Indoor device battery claims",
        "File battery claims for indoor devices within 30 calendar days of discovery. "
        "Include the device ID and the date you discovered the issue.",
    ),
    (
        "Outdoor device battery claims",
        "File battery claims for outdoor devices within 7 calendar days of discovery. "
        "Include the device ID and the date you discovered the issue.",
    ),
    (
        "Replacement kits",
        "Members with an open claim may request a replacement kit at the lending desk. "
        "Kits are issued if stock is available.",
    ),
)


@lru_cache(maxsize=1)
def sample_pdf() -> bytes:
    """Render the same in-memory sample for each session."""
    with pymupdf.open() as document:
        for title, body in _PAGES:
            page = document.new_page()
            page.insert_text((54, 72), title, fontsize=21)
            page.insert_textbox(
                pymupdf.Rect(54, 100, page.rect.width - 54, page.rect.height - 54),
                body,
                fontsize=12,
            )
        return cast(bytes, document.tobytes())
