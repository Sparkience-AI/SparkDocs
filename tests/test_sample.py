from sparkdocs.ingestion import ingest_upload
from sparkdocs.sample import SAMPLE_FILENAME, sample_pdf


def test_local_sample_exposes_both_deadlines_at_distinct_locations() -> None:
    document = ingest_upload(
        filename=SAMPLE_FILENAME,
        claimed_media_type="application/pdf",
        content=sample_pdf(),
    )
    sections = {section.locator: section.text for section in document.sections}

    assert "fictional" in sections["PDF page 1"]
    assert "30 calendar days of discovery" in sections["PDF page 2"]
    assert "7 calendar days of discovery" in sections["PDF page 3"]
    assert "if stock is available" in sections["PDF page 4"]
