"""Streamlit entry point for the SparkDocs alpha."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from sparkdocs.errors import SparkDocsError
from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.providers import GeminiProvider
from sparkdocs.service import DocumentQuestionService

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_POLICY = UploadPolicy()


def clear_session() -> None:
    for key in ("sparkdocs_api_key", "sparkdocs_document", "sparkdocs_upload_digest"):
        st.session_state.pop(key, None)


st.set_page_config(layout="wide", page_title="SparkDocs", page_icon="📄")
st.markdown(
    f"<style>{(APP_ROOT / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.title("SparkDocs")
st.caption("Alpha · local-first bounded document ingestion and experimental grounded Q&A")
st.info(
    "Uploads stay in this Streamlit session. Questions and selected document excerpts are sent "
    "to Google Gemini using your key. Do not upload confidential or regulated documents."
)

with st.sidebar:
    st.subheader("Session controls")
    api_key = st.text_input("Google API key", type="password", key="sparkdocs_api_key")
    if st.button("Clear document and API key", use_container_width=True):
        clear_session()
        st.rerun()
    st.caption(
        "The API key is passed directly to a session-owned client and is not written to disk."
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
            st.error(str(exc))
        else:
            st.session_state.sparkdocs_document = ingested_document
            st.session_state.sparkdocs_upload_digest = digest_key

document = st.session_state.get("sparkdocs_document")
if document is not None:
    st.success(
        f"Ready: {document.kind.upper()} · {len(document.sections)} sections · "
        f"{document.character_count:,} extracted characters"
    )
    st.caption(
        "Extraction is bounded and memory-only. Retrieval is currently basic lexical matching; "
        "answers and citations require user verification."
    )
    question = st.text_input("Question about this document", max_chars=2_000)
    if st.button("Ask SparkDocs", disabled=not bool(api_key and question.strip())):
        try:
            service = DocumentQuestionService(GeminiProvider(api_key))
            with st.spinner("Selecting evidence and generating one grounded answer..."):
                result = service.answer(document, question)
        except (SparkDocsError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The model request failed. Check the key, connection, quota, and try again.")
        else:
            st.markdown(result.answer)
            with st.expander("Evidence locations sent to the model"):
                for locator in result.evidence_locators:
                    st.write(f"- {locator}")
elif uploaded_file is None:
    st.write("Upload a supported document to begin. No shared passkey is required.")
