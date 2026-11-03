"""Streamlit entry point for the SparkDocs alpha."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from sparkdocs.errors import SparkDocsError
from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.providers import (
    DEFAULT_MODELS,
    PROVIDER_LABELS,
    ProviderKind,
    create_provider,
)
from sparkdocs.retrieval import FastEmbedDense, FastEmbedReranker, HybridRetriever, build_index
from sparkdocs.service import DocumentQuestionService

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_POLICY = UploadPolicy()


def clear_session() -> None:
    for key in tuple(st.session_state):
        if key.startswith("sparkdocs_"):
            st.session_state.pop(key, None)


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
st.info(
    "Documents are indexed locally with dense embeddings, BM25, RRF, and a cross-encoder reranker. "
    "The local models download once on first use and are cached on this machine. Only retrieved "
    "excerpts and your question are sent to the cloud model provider you explicitly select."
)

with st.sidebar:
    st.subheader("Session controls")
    provider_kind = ProviderKind(
        st.selectbox(
            "Cloud provider",
            options=list(ProviderKind),
            format_func=lambda value: PROVIDER_LABELS[value],
            key="sparkdocs_provider",
        )
    )
    provider_label = PROVIDER_LABELS[provider_kind]
    model = st.text_input(
        "Model ID",
        value=DEFAULT_MODELS[provider_kind],
        key=f"sparkdocs_model_{provider_kind.value}",
        help="Use a model available to your provider account.",
    )
    api_key = st.text_input(
        f"{provider_label} API key",
        type="password",
        key=f"sparkdocs_api_key_{provider_kind.value}",
    )
    if st.button("Clear document, provider settings, and keys", use_container_width=True):
        clear_session()
        st.rerun()
    st.caption(
        f"Selected route: {provider_label}. The key is passed directly to a session-owned client "
        "and is not written to disk."
    )

uploaded_file = st.file_uploader(
    "Upload one PDF, DOCX, or PPTX document",
    type=["pdf", "docx", "pptx"],
    help="Maximum upload: 10 MiB. Expanded OOXML content and extracted text have separate limits.",
)

if uploaded_file is not None:
    content = uploaded_file.getvalue()
    digest_key = (uploaded_file.name, uploaded_file.type, sha256(content).hexdigest())
    if st.session_state.get("sparkdocs_upload_digest") != digest_key:
        try:
            ingested_document = ingest_upload(
                filename=uploaded_file.name,
                claimed_media_type=uploaded_file.type,
                content=content,
                policy=UPLOAD_POLICY,
            )
        except SparkDocsError as exc:
            st.session_state.pop("sparkdocs_document", None)
            st.session_state.pop("sparkdocs_index", None)
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
    question = st.text_input("Question about this document", max_chars=2_000)
    if st.button("Ask SparkDocs", disabled=not bool(api_key and question.strip())):
        try:
            service = DocumentQuestionService(
                create_provider(provider_kind, api_key, model),
                HybridRetriever(dense_model(), reranker_model()),
            )
            with st.spinner("Running hybrid retrieval, reranking, and grounded generation..."):
                result = service.answer(index, question)
        except (SparkDocsError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The model request failed. Check the key, connection, quota, and try again.")
        else:
            st.markdown(result.answer)
            with st.expander("Evidence locations sent to the model"):
                for locator in result.evidence_locators:
                    marker = "cited" if locator in result.cited_locators else "retrieved"
                    st.write(f"- {locator} ({marker})")
elif uploaded_file is None:
    st.write("Upload a supported document to begin. No shared passkey is required.")
