from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from sparkdocs import providers, retrieval
from sparkdocs.models import Section
from sparkdocs.sample import SAMPLE_FILENAME, sample_pdf

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"
PDF_MIME_TYPE = "application/pdf"


class LocalTestEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any(text.startswith("Document: broken-index.pdf\n") for text in texts):
            raise RuntimeError("Index unavailable")
        return [[1.0] for _ in texts]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    st.cache_resource.clear()
    monkeypatch.setattr(retrieval, "FastEmbedDense", LocalTestEmbedder)
    return AppTest.from_file(str(APP_PATH), default_timeout=15).run()


def upload_sample(app: AppTest, filename: str = SAMPLE_FILENAME) -> None:
    app.file_uploader[0].set_value((filename, sample_pdf(), PDF_MIME_TYPE)).run()
    assert not app.exception


def assert_no_document(app: AppTest) -> None:
    assert not app.exception
    assert "sparkdocs_document" not in app.session_state
    assert "sparkdocs_index" not in app.session_state
    assert "sparkdocs_upload_digest" not in app.session_state
    assert not any(widget.label == "Ask SparkDocs" for widget in app.button)


def test_removing_upload_clears_document_state(app: AppTest) -> None:
    upload_sample(app)
    assert app.session_state["sparkdocs_document"].filename == SAMPLE_FILENAME
    app.session_state["sparkdocs_doc_previous_answer"] = "Old answer"

    app.file_uploader[0].clear().run()

    assert_no_document(app)
    assert "sparkdocs_doc_previous_answer" not in app.session_state


@pytest.mark.parametrize(
    ("failed_filename", "failed_content", "error_hint"),
    [
        ("invalid.pdf", b"Invalid PDF", "signature"),
        ("broken-index.pdf", sample_pdf(), "retrieval models could not be initialized"),
    ],
    ids=["ingestion-failure", "indexing-failure"],
)
def test_failed_replacement_cannot_leave_previous_document_active(
    app: AppTest, failed_filename: str, failed_content: bytes, error_hint: str
) -> None:
    upload_sample(app)
    app.session_state["sparkdocs_doc_previous_answer"] = "Old answer"

    app.file_uploader[0].set_value((failed_filename, failed_content, PDF_MIME_TYPE)).run()

    assert_no_document(app)
    assert "sparkdocs_doc_previous_answer" not in app.session_state
    assert any(error_hint in error.value for error in app.error)

    upload_sample(app, "replacement.pdf")
    assert app.session_state["sparkdocs_document"].filename == "replacement.pdf"


def test_clear_resets_upload_and_provider_widgets(app: AppTest) -> None:
    original_uploader_id = app.file_uploader[0].id
    default_model = app.sidebar.text_input[0].value
    app.sidebar.text_input[0].set_value("different-model")
    app.sidebar.text_input[1].set_value("fake-local-key")
    app.sidebar.checkbox[0].check()
    upload_sample(app)

    app.sidebar.button[0].click().run()

    assert_no_document(app)
    assert app.file_uploader[0].id != original_uploader_id
    assert app.file_uploader[0].value is None
    assert app.sidebar.text_input[0].value == default_model
    assert app.sidebar.text_input[1].value == ""
    assert not app.sidebar.checkbox[0].value


def test_sample_can_be_cleared_back_to_empty_upload(app: AppTest) -> None:
    app.radio[0].set_value("Try the sample handbook").run()

    assert not app.exception
    assert app.session_state["sparkdocs_document"].filename == SAMPLE_FILENAME
    assert not app.file_uploader

    app.sidebar.button[0].click().run()

    assert_no_document(app)
    assert app.radio[0].value == "Upload a document"
    assert app.file_uploader[0].value is None


def test_sample_question_fills_input_without_submitting(app: AppTest) -> None:
    app.radio[0].set_value("Try the sample handbook").run()
    suggestion = next(button for button in app.button if "battery claim" in button.label)
    suggestion.click().run()

    assert not app.exception
    assert (
        app.text_input(
            key=next(
                field.key
                for field in app.text_input
                if field.label == "Question about this document"
            )
        ).value
        == suggestion.label
    )
    assert next(button for button in app.button if button.label == "Ask SparkDocs").disabled
    assert "sparkdocs_doc_view" not in app.session_state


def test_switching_sources_drops_the_previous_upload_and_answer(app: AppTest) -> None:
    upload_sample(app, "my-document.pdf")
    app.session_state["sparkdocs_doc_previous_answer"] = "Old answer"

    app.radio[0].set_value("Try the sample handbook").run()

    assert not app.exception
    assert app.session_state["sparkdocs_document"].filename == SAMPLE_FILENAME
    assert "sparkdocs_doc_previous_answer" not in app.session_state

    app.radio[0].set_value("Upload a document").run()

    assert_no_document(app)
    assert app.file_uploader[0].value is None


def test_switching_result_views_reuses_answer_and_evidence(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = Mock()
    provider.generate.return_value = "Seven days. [PDF page 3]"
    monkeypatch.setattr(providers, "create_provider", Mock(return_value=provider))
    monkeypatch.setattr(retrieval, "FastEmbedReranker", Mock())
    retrieve = Mock(return_value=(Section("PDF page 3", "Seven days."),))
    monkeypatch.setattr(retrieval.HybridRetriever, "retrieve", retrieve)

    app.radio[0].set_value("Try the sample handbook").run()
    next(button for button in app.button if "battery claim" in button.label).click().run()
    app.sidebar.text_input[1].set_value("test-key")
    app.sidebar.checkbox[0].check().run()
    for action in ("Ask SparkDocs", "Find local evidence", "Ask SparkDocs"):
        next(button for button in app.button if button.label == action).click().run()
        assert not app.exception and not app.error
        expected = "Local evidence preview" if action.startswith("Find") else "Answer from"
        assert any(heading.value.startswith(expected) for heading in app.subheader)

    assert provider.generate.call_count == retrieve.call_count == 1
    app.sidebar.button[0].click().run()
    assert "sparkdocs_doc_saved_answer" not in app.session_state
    assert "sparkdocs_doc_saved_evidence" not in app.session_state
