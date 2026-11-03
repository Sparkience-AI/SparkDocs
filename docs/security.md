# Security model

Every upload and every extracted string is untrusted input.

## Implemented controls

- Plain basenames only; path components and hidden names are rejected.
- Extension, declared MIME type, and file signature must agree.
- PDF magic and OOXML package structure are checked before parsing.
- OOXML member paths, encryption flags, member count, individual size, total expanded size, and
  compression ratios are bounded.
- PDF page count, PPTX slide count, normalized section count, and extracted characters are bounded.
- Original bytes remain in memory; no temporary upload path is created.
- A failed evidence search abstains instead of transmitting the entire document.
- Hybrid dense/BM25 retrieval, RRF fusion, and local cross-encoder reranking operate on bounded
  candidates; selected evidence and model responses have separate character limits.
- Document text, locators, and questions are escaped, delimited, and labeled untrusted in the model prompt.
- Model-returned citation labels must match the exact locators of evidence included in that prompt.
- The provider receives no tools and has no filesystem, browser, or mutation authority.

## Known limitations

- PDF and OOXML parsers run in the Streamlit process. There is not yet an operating-system sandbox,
  worker process, hard memory limit, or hard wall-clock timeout.
- Signature and archive checks are not malware scanning or content disarm and reconstruction.
- Retrieval and reranking improve evidence selection but do not guarantee recall. Citation labels are
  allowlisted, but claim-level support and entailment are not independently verified.
- Streamlit session isolation and hosting-platform memory behavior remain part of the deployment trust
  boundary.

Do not use SparkDocs alpha for hostile or high-value documents. Parser isolation, cancellation,
malicious-document evaluation, and claim-level citation verification remain release gates.

Report vulnerabilities privately using the Sparkience security-reporting process. Never attach an
active exploit, private document, or credential to a public issue.
