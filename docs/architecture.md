# Architecture and enforced limits

SparkDocs separates the Streamlit interface from a framework-independent core.

| Layer | Responsibility |
|---|---|
| `ingestion` | Validate names, type claims, signatures, archive structure, and extraction bounds |
| `models` | Represent ephemeral documents and source-located sections |
| `retrieval` | Select at most six lexical matches without an all-document fallback |
| `prompts` | Delimit evidence and enforce grounded, no-tools behavior |
| `providers` | Own the session-scoped Gemini client |
| `service` | Validate questions and coordinate one bounded provider request |
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

These bounds are defense-in-depth, not proof that all accepted files are safe. Wave 3 must move
parsers into disposable bounded workers and validate a citation-aware retrieval system against frozen
synthetic and licensed fixtures.
