# Release checklist

- [ ] Version and changelog describe only supported behavior and known limitations.
- [ ] Lockfile is current; formatting, lint, strict typing, tests, docs/GEO, and Python 3.11–3.13 wheel smoke checks pass.
- [ ] Synthetic PDF/DOCX/PPTX ingestion, retrieval, citation, path-boundary, prompt-injection, provider-failure, and MCP clear-state tests pass.
- [ ] A clean environment installs the wheel and `sparkdocs-mcp --help` works with the MCP extra.
- [ ] Streamlit keyboard navigation, labels, contrast, focus order, error announcements, and narrow viewport are manually checked.
- [ ] The performance envelope is measured with synthetic public fixtures and compared with the previous release.
- [ ] `scripts/release_evidence.py` produces checksums, CycloneDX SBOM, and provenance for the exact source commit.
- [ ] Release notes include privacy boundaries, provider transmission, local model downloads, parser limitations, and rollback instructions.
- [ ] Rollback is rehearsed by reinstalling the previous wheel/lock and confirming its smoke suite.

No Windows installer, executable bundler, signing certificate, or store package is part of the release.
