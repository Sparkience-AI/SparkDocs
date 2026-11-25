"""Streamlit entry point for the SparkDocs alpha."""

import re
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from time import monotonic

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
from sparkdocs.service import (
    AnswerResult,
    DocumentQuestionService,
    answer_general,
    contextualize_question,
    needs_document,
)

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_POLICY = UploadPolicy()
SESSION_GENERATION = "sparkdocs_session_generation"
DOCUMENT_KEYS = {"sparkdocs_document", "sparkdocs_index", "sparkdocs_upload_digest"}
CHAT_HISTORY_KEY = "sparkdocs_doc_chat_history"
MAX_CHAT_TURNS = 20
NO_EVIDENCE_MESSAGE = "No matching evidence was found in the document. Try rephrasing."


@dataclass(frozen=True)
class ChatTurn:
    question: str
    response: AnswerResult | tuple[Section, ...] | str
    provider: ProviderKind | None = None


def render_response(turn: ChatTurn, *, animate: bool = False) -> None:
    response = turn.response
    if turn.provider is not None and isinstance(response, (AnswerResult, str)):
        if isinstance(response, AnswerResult):
            st.caption("Document answer")
        else:
            st.caption("General answer · No document excerpts sent")
        text = response.answer if isinstance(response, AnswerResult) else response
        safe_answer = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escape(text))
        appearance = " sparkdocs-new-answer" if animate else ""
        st.html(f'<div class="sparkdocs-answer{appearance}">{safe_answer}</div>')
        if isinstance(response, str):
            return
        provider_name = PROVIDER_LABELS[turn.provider]
        cited_locators = response.cited_locators
        for excerpt in response.evidence:
            if excerpt.locator in cited_locators:
                with st.expander(f"{excerpt.locator} (cited)", icon=":material/verified:"):
                    st.text(excerpt.text)
        other_excerpts = [item for item in response.evidence if item.locator not in cited_locators]
        if other_excerpts:
            with st.expander(
                f"Other passages sent to {provider_name} ({len(other_excerpts)})",
                icon=":material/folder_open:",
            ):
                for excerpt in other_excerpts:
                    st.text(excerpt.locator)
                    st.text(excerpt.text)
    elif isinstance(response, tuple):
        st.caption("Local evidence preview · no cloud request")
        if response:
            for excerpt in response:
                with st.expander(f"{excerpt.locator} (retrieved)"):
                    st.text(excerpt.text)
        else:
            st.info(NO_EVIDENCE_MESSAGE)


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

st.html(
    '<header class="sparkdocs-brand"><span class="sparkdocs-mark" aria-hidden="true">'
    "✦</span><h1>SparkDocs</h1></header>"
)

generation = st.session_state.get(SESSION_GENERATION, 0)
with st.sidebar:
    st.subheader("Settings")
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
        f"Send my messages and limited chat context to {provider_label}. "
        "Document answers also send matching excerpts.",
        key=f"sparkdocs_consent_{provider_kind.value}_{generation}",
    )
    st.button(
        "Clear document, provider settings, and keys",
        width="stretch",
        on_click=clear_session,
    )
    st.caption(
        f"Selected route: {provider_label}. The key is passed directly to a session-owned client "
        "and is not written to disk. Chat history is stored in this session."
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
    if st.session_state.get("sparkdocs_upload_digest") is not None:
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
    display_name = (
        "Maple claims handbook" if source == "Try the sample handbook" else document.filename
    )
    st.html(
        '<div class="sparkdocs-document"><span class="sparkdocs-ready">Ready</span>'
        f"<strong>{escape(display_name)}</strong><span>{document.kind.upper()} · "
        f"{len(document.sections)} sections</span></div>"
    )


def render_chat(mode: str, slug: str, *, available: bool) -> None:
    question_key = f"sparkdocs_doc_question_{slug}"
    history: list[ChatTurn] = st.session_state.setdefault(f"{CHAT_HISTORY_KEY}_{slug}", [])
    if not available:
        st.caption("Upload a document or choose the sample handbook to start here.")
    if (
        available
        and index is not None
        and slug != "general"
        and source == "Try the sample handbook"
    ):
        with st.container(horizontal=True):
            for label, sample_question in zip(
                ("Outdoor claim deadline", "Replacement kit limit"), SAMPLE_QUESTIONS, strict=True
            ):
                st.button(
                    label,
                    key=f"sparkdocs_doc_sample_{slug}_{label}",
                    on_click=st.session_state.__setitem__,
                    args=(question_key, sample_question),
                )
    conversation = st.container()
    with conversation:
        if not history:
            intro = st.empty()
            intro.html(f'<div class="sparkdocs-intro"><h2>{mode}</h2></div>')
        for turn in history:
            with st.chat_message("user", avatar=":material/person:"):
                st.html(f'<div class="sparkdocs-user">{escape(turn.question)}</div>')
            with st.chat_message("assistant", avatar=":material/description:"):
                render_response(turn)

    ask_cloud = mode != "Find local evidence"
    can_ask_cloud = bool(api_key.strip() and cloud_consent and model.strip())
    if ask_cloud and not can_ask_cloud:
        st.caption("Set a model ID and API key and confirm cloud consent in the sidebar.")
    question = (
        st.bottom.chat_input(
            "Message SparkDocs",
            max_chars=2_000,
            key=question_key,
            disabled=not available or (ask_cloud and not can_ask_cloud),
            submit_mode="disable",
        )
        or ""
    ).strip()
    if available and question and (not ask_cloud or can_ask_cloud):
        if not history:
            intro.empty()
        with conversation, st.chat_message("user", avatar=":material/person:"):
            st.html(f'<div class="sparkdocs-user">{escape(question)}</div>')
        with conversation, st.chat_message("assistant", avatar=":material/description:"):
            preview = st.empty()
            chunks: list[str] = []
            last_draw = 0.0

            def show_delta(delta: str) -> None:
                nonlocal last_draw
                chunks.append(delta)
                now = monotonic()
                if now - last_draw >= 0.05:
                    preview.text("".join(chunks))
                    last_draw = now

            try:
                # Reuse the same evidence when inspecting the immediately preceding question.
                if not ask_cloud and history and history[-1].question == question:
                    previous_response = history[-1].response
                    evidence = (
                        previous_response.evidence
                        if isinstance(previous_response, AnswerResult)
                        else previous_response
                    )
                    turn = ChatTurn(question, evidence)
                elif ask_cloud:
                    provider = create_provider(provider_kind, api_key, model)
                    previous_questions = [
                        previous.question
                        for previous in history
                        if previous.provider == provider_kind
                        and isinstance(previous.response, AnswerResult)
                    ]
                    with st.spinner("Choosing how to answer..."):
                        use_document = index is not None and (
                            mode == "Document only"
                            or (
                                mode == "Ask SparkDocs"
                                and needs_document(provider, question, previous_questions)
                            )
                        )
                    if use_document:
                        service = DocumentQuestionService(
                            provider, HybridRetriever(dense_model(), reranker_model())
                        )
                        with st.spinner("Searching sources and checking citations..."):
                            result = service.answer(
                                index,
                                question,
                                previous_questions=previous_questions,
                                on_delta=show_delta,
                            )
                        turn = ChatTurn(question, result, provider_kind)
                    else:
                        exchanges = [
                            (previous.question, previous.response)
                            for previous in history
                            if previous.provider == provider_kind
                            and isinstance(previous.response, str)
                        ]
                        with st.spinner("Writing an answer..."):
                            response = answer_general(provider, question, exchanges, show_delta)
                        turn = ChatTurn(question, response, provider_kind)
                else:
                    previous_questions = [previous.question for previous in history]
                    with st.spinner("Finding local evidence..."):
                        retriever = HybridRetriever(dense_model(), reranker_model())
                        evidence = retriever.retrieve(
                            index, contextualize_question(question, previous_questions)
                        )
                    turn = ChatTurn(question, evidence)
            except InsufficientEvidenceError:
                st.info(NO_EVIDENCE_MESSAGE)
            except SparkDocsError as exc:
                st.error(str(exc))
            except Exception:
                if ask_cloud:
                    st.error(
                        "Could not answer the question. Check your model settings and try again."
                    )
                else:
                    st.error("Could not retrieve local evidence. Check your models and try again.")
            else:
                history.append(turn)
                if len(history) > MAX_CHAT_TURNS:
                    del history[:-MAX_CHAT_TURNS]
                    st.rerun()
                render_response(turn, animate=True)
            finally:
                preview.empty()


chat_modes = (
    ("Ask SparkDocs", "ask"),
    ("Document only", "document"),
    ("General chat", "general"),
    ("Find local evidence", "evidence"),
)
with st.container(key="sparkdocs_chat_panel"):
    tabs = st.tabs(
        [label for label, _ in chat_modes], key="sparkdocs_active_tab", on_change="rerun"
    )
    for tab, (label, slug) in zip(tabs, chat_modes, strict=True):
        if not tab.open:
            continue
        with tab:
            render_chat(label, slug, available=slug in {"ask", "general"} or index is not None)
