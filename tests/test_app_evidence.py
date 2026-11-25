from collections.abc import Sequence
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from sparkdocs import providers, retrieval
from sparkdocs.sample import SAMPLE_QUESTIONS

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class FakeReranker:
    def score(self, question: str, documents: Sequence[str]) -> list[float]:
        return [float("Outdoor device battery claims" in text) for text in documents]


def test_sample_evidence_is_visible_without_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_cloud_call(*args: object, **kwargs: object) -> None:
        pytest.fail("Local preview tried to construct a cloud provider.")

    st.cache_resource.clear()
    monkeypatch.setattr(retrieval, "FastEmbedDense", FakeEmbedder)
    monkeypatch.setattr(retrieval, "FastEmbedReranker", FakeReranker)
    monkeypatch.setattr(providers, "create_provider", reject_cloud_call)
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.radio[0].set_value("Try the sample handbook").run()
    app.session_state["sparkdocs_active_tab"] = "Find local evidence"
    app.run()
    app.chat_input[0].set_value(SAMPLE_QUESTIONS[0]).run()

    assert not app.exception
    page = next(panel for panel in app.expander if panel.label == "PDF page 3 (retrieved)")
    assert "within 7 calendar days of discovery" in page.text[0].value
