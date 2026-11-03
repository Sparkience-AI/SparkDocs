# Architecture and enforced limits

SparkDocs separates the Streamlit interface from a framework-independent core.

| Layer | Responsibility |
|---|---|
| `ingestion` | Validate names, type claims, signatures, archive structure, and extraction bounds |
| `models` | Represent ephemeral documents, source-located chunks, and in-memory indexes |
| `chunking` | Split within source sections with overlap while preserving human-readable locators |
| `retrieval` | Fuse local dense and BM25 candidates with RRF, rerank, and enforce a context budget |
| `prompts` | Delimit evidence and enforce grounded, no-tools behavior |
| `providers` | Own session-scoped Gemini, OpenAI Responses, and Anthropic Messages clients |
| `service` | Coordinate one bounded provider request and validate returned citation labels |
| `code.py` | Render the UI and translate product-level errors |

Default ingestion limits:

| Limit | Default |
|---|---:|
| Upload bytes | 10 MiB |
| Expanded OOXML bytes | 40 MiB |
| OOXML members | 2,000 |
| Single OOXML member | 10 MiB |
| Compression ratio | 100:1 |
| PDF pages | 200 |
| PPTX slides | 200 |
| Extracted sections | 1,000 |
| Extracted characters | 1,000,000 |
| Question characters | 2,000 |

The retrieval index and document content remain in process memory. FastEmbed model weights are
downloaded on first use and may remain in the machine or hosting account's model cache, but user
documents are not written into that cache. Dense retrieval uses `BAAI/bge-small-en-v1.5`; the query
and passage embedding paths provided by FastEmbed are applied.
`Xenova/ms-marco-MiniLM-L-6-v2` reranks a bounded candidate set. Normal tests use deterministic
fakes and never download these models.

These bounds are defense-in-depth, not proof that all accepted files are safe. Parser isolation,
claim-level citation verification, and broader retrieval evaluation remain release work.
