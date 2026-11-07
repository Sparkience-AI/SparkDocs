# SparkDocs

**Status: alpha.** SparkDocs is a local-first document ingestion and grounded Q&A
tool for PDF, DOCX, and PPTX files. It is intended for people who want to inspect ordinary
documents with an explicit bring-your-own-key model rather than upload them to a managed document
store.

SparkDocs validates and extracts documents in memory, builds structure-aware overlapping chunks,
and retrieves evidence locally using dense embeddings plus BM25 keyword search. Reciprocal rank
fusion (RRF) combines those results, a local cross-encoder reranks them, and only a bounded evidence
set is sent in one request to the user-selected Gemini, OpenAI, or Anthropic model. Returned citation
labels are restricted to retrieved locations;
SparkDocs does not independently prove that every generated claim is entailed by its citation.

## What it does

- Accepts one PDF, DOCX, or PPTX document up to 10 MiB.
- Checks filename, extension, media type, file signature, and OOXML archive structure.
- Limits archive members, expanded size, compression ratio, pages/slides, sections, and extracted
  characters.
- Preserves PDF page, DOCX paragraph/table-row, and PPTX slide locators.
- Uses local FastEmbed ONNX models for dense retrieval and cross-encoder reranking.
- Applies a strict evidence budget and rejects citations outside the retrieved-location allowlist.
- Keeps uploads and extracted content in the active Streamlit process memory only.
- Sends only character-bounded selected evidence and the escaped question to the explicitly
  selected Gemini, OpenAI, or Anthropic model using a session-only user key.
- Treats document content as untrusted data and does not give it tools or external authority.

## What it does not do

- It does not authenticate users. The former public shared passkey has been removed.
- It does not persist documents, indexes, questions, answers, or API keys.
- It does not support scanned/image-only documents, OCR, encrypted files, macros, or arbitrary
  archive formats.
- It does not guarantee answer correctness, completeness, citation accuracy, or protection suitable
  for confidential, regulated, or adversarial documents.
- It does not expose a network MCP service, arbitrary filesystem access, model-authored code, OCR,
  or document editing.

## Local installation

Prerequisites:

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/)
- A Gemini, OpenAI, or Anthropic API key for Q&A; ingestion and tests do not need a key
- Network access on first interactive use to download the local embedding and reranking model
  weights; subsequent runs use the local model cache

```bash
git clone https://github.com/Sparkience-AI/SparkDocs.git
cd SparkDocs
uv sync --locked
uv run --locked streamlit run code.py
```

Open <http://127.0.0.1:8501>, select Gemini, OpenAI, or Anthropic, confirm the model ID, enter the
matching API key, explicitly confirm cloud transmission, and upload one supported document. The cost-conscious defaults are
`gemini-3.8-flash`, `gpt-6-luna`, and `claude-haiku-5-5`; model availability depends on the user's
provider account and the field is editable. The committed local configuration binds Streamlit to
loopback and disables Streamlit usage reporting. Keys are passed directly to session-owned clients;
they are not copied into the process environment or written to disk. OpenAI requests explicitly set
`store=false`. FastEmbed downloads model weights from Hugging Face on first use and caches them
locally; document text is not sent for embedding or reranking. A hosted deployment must explicitly
override the local server address and should retain the restricted demo contract below.

## Hosted demo

The historical demo URL is <https://sparkdocs.streamlit.app/>. Treat it as an optional demonstration,
not the complete-product architecture. Deployment health and parity with the current branch are not
guaranteed until a reviewed release is deployed. A hosted session receives both the user's document
and API key in server memory, so use only disposable low-quota keys and non-sensitive documents.

## Security and privacy

Read the [privacy contract](docs/privacy.md), [security model](docs/security.md), and
[architecture and limits](docs/architecture.md) before processing documents. Important boundaries:

- The selected cloud provider receives the question and selected excerpts when Q&A is used.
- Filename and archive validation reduce common ingestion risks but do not replace malware scanning
  or parser isolation.
- Parser execution is currently in-process and has no operating-system sandbox or hard wall-clock
  timeout. This is a known alpha limitation.
- No prompt, document text, answer, key, or filename should be logged or included in test artifacts.

## Development

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv build --no-sources
```

The tests use generated synthetic documents and a fake model provider; normal CI does not require a
credential, network request, or real user document.

## Local MCP and agent skill

Install the optional MCP dependency and start the stdio server with an intentionally narrow root:

```bash
uv sync --locked --extra mcp
uv run --locked --extra mcp sparkdocs-mcp --allowed-root ./documents
```

The server holds one document in memory and exposes four tools: `ingest_document`,
`search_document`, `document_status`, and `clear_document`. It never accepts an API key, generates
an answer, opens a network listener, or reads outside the configured root. First indexing can still
download the documented local FastEmbed weights.

The repo-local skill at `.agents/skills/sparkdocs-grounded-qa` teaches compatible agents to ingest,
retrieve, cite exact locators, resist document prompt injection, and abstain when evidence is
insufficient. The MCP is the controlled evidence surface; the skill is the workflow layer.

See the [threat model](docs/threat-model.md), [performance envelope](docs/performance.md), and
[release checklist](RELEASE_CHECKLIST.md) for release gates.

## Architecture

```text
Streamlit UI
    │
    ├── bounded upload validation and extraction
    │       └── normalized document sections with source locators
    │
    ├── structure-aware overlapping chunks
    │       └── local dense + BM25 retrieval → RRF → cross-encoder reranking
    │
    └── bounded, escaped evidence prompt
            └── one grounded provider request + citation allowlist validation
```

The shared core lives under `src/sparkdocs`; the current Streamlit UI remains the source-checkout
entry point. The built wheel is the reusable core package rather than a complete desktop or server
distribution. Future UI, MCP, or agent surfaces must call the same core rather than duplicate
parsing or security logic.

## License and support

SparkDocs is available under the [MIT License](LICENSE). Use GitHub issues for reproducible bugs and
feature proposals. Do not post real documents, credentials, or confidential content in an issue.
