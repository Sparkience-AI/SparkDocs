# Performance envelope

The supported envelope is one in-memory document per process: at most 10 MiB compressed input, 40 MiB expanded OOXML, 200 PDF pages or PPTX slides, 1,000 extracted sections, and 1,000,000 extracted characters. MCP search returns at most six passages and 12,000 evidence characters; the Streamlit answer path allows 24,000 evidence characters.

First use can download embedding and reranking weights and is intentionally excluded from steady-state latency. Hardware, model cache, document structure, and provider latency make a universal time guarantee misleading. Release verification must record platform, cold/warm cache state, input size, extraction time, indexing time, peak resident memory where available, and query time. A release is blocked by unbounded growth, limit bypass, or a regression above 2× the prior recorded synthetic-fixture median without explanation.
