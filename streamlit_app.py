"""Streamlit entry point for the SparkDocs alpha."""

import re
from hashlib import sha256
from html import escape
from pathlib import Path

import streamlit as st

from sparkdocs.errors import InsufficientEvidenceError, SparkDocsError
from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.models import Section
from sparkdocs.providers import (
    DEFAULT_MODELS,
    PROVIDER_LABELS,
    ProviderKind,
    create_provider,
)
from sparkdocs.retrieval import FastEmbedDense, FastEmbedReranker, HybridRetriever, build_index
from sparkdocs.sample import SAMPLE_FILENAME, SAMPLE_QUESTIONS, sample_pdf
from sparkdocs.service import AnswerResult, DocumentQuestionService

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_POLICY = UploadPolicy()
SESSION_GENERATION = "sparkdocs_session_generation"
DOCUMENT_KEYS = {"sparkdocs_document", "sparkdocs_index", "sparkdocs_upload_digest"}


def clear_document() -> None:
    for key in tuple(st.session_state):
        if key in DOCUMENT_KEYS or key.startswith("sparkdocs_doc_"):
            st.session_state.pop(key, None)


def clear_session() -> None:
    generation = st.session_state.get(SESSION_GENERATION, 0)
    for key in tuple(st.session_state):
        if key.startswith("sparkdocs_"):
            st.session_state.pop(key, None)
    # A new widget identity clears the upload and sidebar fields in the browser as well.
    st.session_state[SESSION_GENERATION] = generation + 1


@st.cache_resource(show_spinner=False)
def dense_model() -> FastEmbedDense:
    return FastEmbedDense()


@st.cache_resource(show_spinner=False)
def reranker_model() -> FastEmbedReranker:
    return FastEmbedReranker()


st.set_page_config(layout="wide", page_title="SparkDocs", page_icon="📄")
st.markdown(
    f"<style>{(APP_ROOT / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.title("SparkDocs")
st.caption("Alpha · local-first hybrid RAG with grounded, source-located citations")
st.info("Evidence search runs locally. Cloud answers send your question and retrieved excerpts.")

generation = st.session_state.get(SESSION_GENERATION, 0)
with st.sidebar:
    st.subheader("Session controls")
    provider_kind = ProviderKind(
        st.selectbox(
            "Cloud provider",
            options=list(ProviderKind),
            format_func=lambda value: PROVIDER_LABELS[value],
            key=f"sparkdocs_provider_{generation}",
        )
    )
    provider_label = PROVIDER_LABELS[provider_kind]
    model = st.text_input(
        "Model ID",
        value=DEFAULT_MODELS[provider_kind],
        key=f"sparkdocs_model_{provider_kind.value}_{generation}",
        help="Use a model available to your provider account.",
    )
    api_key = st.text_input(
        f"{provider_label} API key",
        type="password",
        key=f"sparkdocs_api_key_{provider_kind.value}_{generation}",
    )
    cloud_consent = st.checkbox(
        f"Send the question and retrieved excerpts to {provider_label}",
        key=f"sparkdocs_consent_{provider_kind.value}_{generation}",
    )
    st.button(
        "Clear document, provider settings, and keys",
        use_container_width=True,
        on_click=clear_session,
    )
    st.caption(
        f"Selected route: {provider_label}. The key is passed directly to a session-owned client "
        "and is not written to disk. Q&A remains disabled until cloud transmission is confirmed."
    )

source = st.radio(
    "Document source",
    ["Upload a document", "Try the sample handbook"],
    horizontal=True,
    key=f"sparkdocs_source_{generation}",
)
if source == "Try the sample handbook":
    upload = (SAMPLE_FILENAME, "application/pdf", sample_pdf())
else:
    uploaded_file = st.file_uploader(
        "Upload one PDF, DOCX, or PPTX document",
        type=["pdf", "docx", "pptx"],
        help="Maximum upload: 10 MiB. Expanded OOXML and extracted text have separate limits.",
        key=f"sparkdocs_upload_{generation}",
    )
    upload = (
        (uploaded_file.name, uploaded_file.type, uploaded_file.getvalue())
        if uploaded_file is not None
        else None
    )

if upload is None:
    clear_document()
else:
    filename, media_type, content = upload
    digest_key = (filename, media_type, sha256(content).hexdigest())
    if st.session_state.get("sparkdocs_upload_digest") != digest_key:
        clear_document()
        try:
            ingested_document = ingest_upload(
                filename=filename,
                claimed_media_type=media_type,
                content=content,
                policy=UPLOAD_POLICY,
            )
        except SparkDocsError as exc:
            st.error(str(exc))
        else:
            try:
                with st.spinner("Building the local hybrid retrieval index..."):
                    index = build_index(ingested_document, dense_model())
            except Exception:
                st.error(
                    "The local retrieval models could not be initialized. Check network access "
                    "for the first model download, available disk space, and try again."
                )
            else:
                st.session_state.sparkdocs_document = ingested_document
                st.session_state.sparkdocs_index = index
                st.session_state.sparkdocs_upload_digest = digest_key

document = st.session_state.get("sparkdocs_document")
index = st.session_state.get("sparkdocs_index")
if document is not None and index is not None:
    st.success(
        f"Ready: {document.kind.upper()} · {len(document.sections)} sections · "
        f"{document.character_count:,} extracted characters"
    )
    st.caption(
        f"Indexed {len(index.chunks)} structure-aware chunks in memory. Answers must cite only "
        "retrieved locations; citations still require user verification."
    )
    question_key = f"sparkdocs_doc_question_{document.sha256}"
    if source == "Try the sample handbook":
        st.caption("Choose a sample question to fill the field below.")
        with st.container(horizontal=True):
            for sample_question in SAMPLE_QUESTIONS:
                st.button(
                    sample_question,
                    on_click=st.session_state.__setitem__,
                    args=(question_key, sample_question),
                )
    question = st.text_input(
        "Question about this document",
        max_chars=2_000,
        key=question_key,
    ).strip()
    can_ask = bool(api_key.strip() and cloud_consent and question)
    with st.container(horizontal=True):
        ask_sparkdocs = st.button("Ask SparkDocs", type="primary", disabled=not can_ask)
        find_evidence = st.button("Find local evidence", disabled=not question)
    view_key = "sparkdocs_doc_view"
    no_evidence_message = "No matching evidence was found in the document. Try rephrasing."
    if find_evidence or ask_sparkdocs:
        st.session_state.pop(view_key, None)
        try:
            with st.spinner("Preparing local retrieval..."):
                retriever = HybridRetriever(dense_model(), reranker_model())
        except Exception:
            st.error(
                "The local retrieval models could not be initialized. Check disk space and "
                "network access for the first model download."
            )
            st.stop()

        if ask_sparkdocs and not model.strip():
            st.error("Enter a model ID available to your provider account.")
            st.stop()

        try:
            if find_evidence:
                with st.spinner("Finding evidence on the machine running SparkDocs..."):
                    evidence = retriever.retrieve(index, question)
                if evidence:
                    st.session_state[view_key] = (question, evidence, None)
                else:
                    st.info(no_evidence_message)
            else:
                service = DocumentQuestionService(
                    create_provider(provider_kind, api_key, model), retriever
                )
                with st.spinner("Running hybrid retrieval, reranking, and grounded generation..."):
                    result = service.answer(index, question)
                st.session_state[view_key] = (question, result, provider_label)
        except InsufficientEvidenceError:
            st.info(no_evidence_message)
        except SparkDocsError as exc:
            st.error(str(exc))
        except Exception:
            if find_evidence:
                st.error("Local evidence retrieval failed. Try again or reload the document.")
            else:
                st.error("Could not complete the question. Check the model settings and try again.")

    view = st.session_state.get(view_key)
    if view is not None and view[0] == question:
        _, view_result, answer_provider = view
        excerpts: tuple[Section, ...]
        if isinstance(view_result, AnswerResult):
            st.subheader(f"Answer from {answer_provider}")
            safe_answer = re.sub(
                r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escape(view_result.answer)
            )
            st.html(
                f'<div style="white-space: pre-wrap; overflow-wrap: anywhere">{safe_answer}</div>'
            )
            st.subheader("Evidence sent to the model")
            cited_locators = view_result.cited_locators
            for excerpt in view_result.evidence:
                if excerpt.locator in cited_locators:
                    with st.expander(
                        f"{excerpt.locator} (cited)", expanded=len(cited_locators) == 1
                    ):
                        st.text(excerpt.text)
            other_excerpts = [
                item for item in view_result.evidence if item.locator not in cited_locators
            ]
            if other_excerpts:
                with st.expander(
                    f"Other passages sent to {answer_provider} ({len(other_excerpts)})"
                ):
                    for excerpt in other_excerpts:
                        st.text(excerpt.locator)
                        st.text(excerpt.text)
        else:
            st.subheader("Local evidence preview")
            st.caption("SparkDocs did not send these excerpts to a model provider.")
            excerpts = view_result
            for excerpt in excerpts:
                with st.expander(f"{excerpt.locator} (retrieved)"):
                    st.text(excerpt.text)
elif upload is None:
    st.write("Upload a supported document to begin. No shared passkey is required.")
