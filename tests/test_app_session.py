from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from sparkdocs import providers, retrieval
from sparkdocs.models import Section
from sparkdocs.sample import SAMPLE_FILENAME, SAMPLE_QUESTIONS, sample_pdf

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"
PDF_MIME_TYPE = "application/pdf"


@pytest.mark.parametrize("load_document", [False, True])
def test_general_chat_without_retrieval(
    app: AppTest, monkeypatch: pytest.MonkeyPatch, load_document: bool
) -> None:
    provider = Mock()
    provider.generate.side_effect = ["GENERAL", "Paris."] if load_document else ["Paris."]
    monkeypatch.setattr(providers, "create_provider", Mock(return_value=provider))
    retrieve = Mock()
    monkeypatch.setattr(retrieval.HybridRetriever, "retrieve", retrieve)
    if load_document:
        app.radio[0].set_value("Try the sample handbook").run()
    app.sidebar.text_input[1].set_value("test-key")
    app.sidebar.checkbox[0].check().run()
    app.chat_input[0].set_value("What is the capital of France?").run()
    assert not app.error and not app.exception
    assert any("General answer" in caption.value for caption in app.caption)
    assert not app.expander
    assert provider.generate.call_count == (2 if load_document else 1)
    assert all("Maple" not in call.args[0] for call in provider.generate.call_args_list)
    retrieve.assert_not_called()


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
    assert [tab.label for tab in app.tabs] == [
        "Ask SparkDocs",
        "Document only",
        "General chat",
        "Find local evidence",
    ]


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
    suggestion = next(button for button in app.button if button.label == "Outdoor claim deadline")
    suggestion.click().run()

    assert not app.exception
    assert app.chat_input[0].proto.value == SAMPLE_QUESTIONS[0]
    assert not app.chat_message
    assert not app.session_state["sparkdocs_doc_chat_history_ask"]


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


def test_chat_keeps_followups_and_citations_across_local_mode(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = Mock()
    provider.generate.side_effect = ["Seven days. [PDF page 3]", "Thirty days. [PDF page 2]"]
    monkeypatch.setattr(providers, "create_provider", Mock(return_value=provider))
    monkeypatch.setattr(retrieval, "FastEmbedReranker", Mock())
    outdoor = Section("PDF page 3", "Seven days.")
    indoor = Section("PDF page 2", "Thirty days.")
    retrieve = Mock(side_effect=[(outdoor,), (outdoor, indoor), (indoor,)])
    monkeypatch.setattr(retrieval.HybridRetriever, "retrieve", retrieve)

    app.radio[0].set_value("Try the sample handbook").run()
    app.session_state["sparkdocs_active_tab"] = "Document only"
    app.run()
    assert app.chat_input[0].proto.disabled
    assert provider.generate.call_count == 0

    app.sidebar.text_input[1].set_value("test-key")
    app.sidebar.checkbox[0].check().run()
    first = "When is the outdoor battery claim due?"
    app.session_state["sparkdocs_active_tab"] = "Document only"
    app.run()
    app.chat_input[0].set_value(first).run()
    assert not app.exception and not app.error
    assert [item.label for item in app.expander] == ["PDF page 3 (cited)"]
    assert not app.expander[0].proto.expanded
    assert len(app.chat_message) == 2

    assert provider.generate.call_count == 1
    app.session_state["sparkdocs_active_tab"] = "Find local evidence"
    app.run()
    assert not app.chat_message
    assert len(app.chat_input) == 1
    app.chat_input[0].set_value("Local-only comparison of both rules").run()
    assert len(app.chat_message) == 2
    assert provider.generate.call_count == 1

    followup = "What about indoor ones?"
    app.session_state["sparkdocs_active_tab"] = "Document only"
    app.run()
    assert len(app.chat_message) == 2
    app.chat_input[0].set_value(followup).run()
    assert not app.exception and not app.error
    assert len(app.chat_message) == 4
    assert [item.label for item in app.expander if "(cited)" in item.label] == [
        "PDF page 3 (cited)",
        "PDF page 2 (cited)",
    ]
    assert retrieve.call_count == 3
    assert first in retrieve.call_args.args[1]
    prompt = provider.generate.call_args.args[0]
    assert first in prompt and followup in prompt
    assert "Local-only comparison" not in prompt
    assert provider.generate.call_count == 2
    app.run()
    assert provider.generate.call_count == 2
    assert len(app.chat_message) == 4

    app.radio[0].set_value("Upload a document").run()
    assert not app.chat_message
    key = "sparkdocs_doc_chat_history_document"
    assert key not in app.session_state or not app.session_state[key]
