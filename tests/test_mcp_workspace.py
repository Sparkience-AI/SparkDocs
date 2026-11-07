from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document

from sparkdocs.mcp_workspace import DocumentWorkspace, WorkspaceError


class KeywordEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if "retention" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        return [1.0, 0.0] if "retention" in text.lower() else [0.0, 1.0]


class StableReranker:
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [float(len(documents) - index) for index in range(len(documents))]


class FailingEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError("internal-path-or-secret")

    def embed_query(self, text: str) -> Sequence[float]:
        raise RuntimeError("internal-path-or-secret")


def _write_docx(path: Path) -> None:
    output = BytesIO()
    document = Document()
    document.add_paragraph("The retention period is thirty days.")
    document.save(output)
    path.write_bytes(output.getvalue())


def _workspace(root: Path) -> DocumentWorkspace:
    return DocumentWorkspace(
        root,
        embedder_factory=KeywordEmbedder,
        reranker_factory=StableReranker,
    )


def test_workspace_ingests_searches_and_clears_without_provider(tmp_path: Path) -> None:
    _write_docx(tmp_path / "policy.docx")
    workspace = _workspace(tmp_path)

    status = workspace.ingest("policy.docx")
    result = workspace.search("What is the retention period?")

    assert status.loaded is True
    assert status.kind == "docx"
    assert result["status"] == "evidence_found"
    assert result["evidence"] == [
        {"locator": "DOCX paragraph 1", "text": "The retention period is thirty days."}
    ]
    assert workspace.clear().loaded is False


@pytest.mark.parametrize(
    "path", ["../outside.docx", "C:/outside.docx", "/outside.docx", "missing.docx"]
)
def test_workspace_rejects_paths_outside_root_or_missing(tmp_path: Path, path: str) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="relative|outside|does not exist"):
        workspace.ingest(path)


def test_workspace_requires_ingestion_and_bounds_questions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="Ingest"):
        workspace.search("retention")
    with pytest.raises(WorkspaceError, match="2,000"):
        workspace.search("x" * 2_001)


def test_workspace_status_does_not_expose_path_or_filename(tmp_path: Path) -> None:
    _write_docx(tmp_path / "confidential-name.docx")
    status = _workspace(tmp_path).ingest("confidential-name.docx")

    rendered = repr(status)
    assert str(tmp_path) not in rendered
    assert "confidential-name" not in rendered


def test_workspace_sanitizes_unexpected_indexing_errors(tmp_path: Path) -> None:
    _write_docx(tmp_path / "policy.docx")
    workspace = DocumentWorkspace(
        tmp_path,
        embedder_factory=FailingEmbedder,
        reranker_factory=StableReranker,
    )

    with pytest.raises(WorkspaceError, match="indexed safely") as caught:
        workspace.ingest("policy.docx")

    assert "internal-path-or-secret" not in str(caught.value)
