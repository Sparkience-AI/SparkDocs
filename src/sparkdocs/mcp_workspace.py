"""Local-only, bounded workspace used by the SparkDocs MCP surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sparkdocs.errors import SparkDocsError
from sparkdocs.ingestion import FORMAT_SPECS, UploadPolicy, ingest_upload
from sparkdocs.models import DocumentIndex, IngestedDocument
from sparkdocs.retrieval import (
    DenseEmbedder,
    FastEmbedDense,
    FastEmbedReranker,
    HybridRetriever,
    Reranker,
    build_index,
)


class WorkspaceError(ValueError):
    """A stable, non-secret-bearing MCP workspace error."""


@dataclass(frozen=True, slots=True)
class WorkspaceStatus:
    loaded: bool
    document_sha256: str | None = None
    kind: str | None = None
    sections: int = 0
    chunks: int = 0
    characters: int = 0


class DocumentWorkspace:
    """One in-memory document constrained to a configured local directory."""

    def __init__(
        self,
        allowed_root: Path,
        *,
        policy: UploadPolicy | None = None,
        embedder_factory: Callable[[], DenseEmbedder] = FastEmbedDense,
        reranker_factory: Callable[[], Reranker] = FastEmbedReranker,
    ) -> None:
        try:
            root = allowed_root.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceError("The configured document directory does not exist.") from exc
        if not root.is_dir():
            raise WorkspaceError("The configured document root must be a directory.")
        self._root = root
        self._policy = policy or UploadPolicy()
        self._embedder_factory = embedder_factory
        self._reranker_factory = reranker_factory
        self._document: IngestedDocument | None = None
        self._index: DocumentIndex | None = None
        self._retriever: HybridRetriever | None = None

    def _resolve_document(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if not relative_path.strip() or candidate.is_absolute():
            raise WorkspaceError("Document path must be relative to the configured root.")
        try:
            resolved = (self._root / candidate).resolve(strict=True)
        except OSError as exc:
            raise WorkspaceError("The requested document does not exist.") from exc
        if not resolved.is_relative_to(self._root) or not resolved.is_file():
            raise WorkspaceError("Document path is outside the configured root or is not a file.")
        return resolved

    def ingest(self, relative_path: str) -> WorkspaceStatus:
        path = self._resolve_document(relative_path)
        spec = FORMAT_SPECS.get(path.suffix.lower())
        if spec is None:
            raise WorkspaceError("Only PDF, DOCX, and PPTX documents are supported.")
        try:
            if path.stat().st_size > self._policy.max_upload_bytes:
                raise WorkspaceError("The requested document exceeds the upload limit.")
            payload = path.read_bytes()
        except WorkspaceError:
            raise
        except OSError as exc:
            raise WorkspaceError("The requested document could not be read.") from exc
        if len(payload) > self._policy.max_upload_bytes:
            raise WorkspaceError("The requested document exceeds the upload limit.")
        try:
            document = ingest_upload(
                filename=path.name,
                claimed_media_type=spec.media_type,
                content=payload,
                policy=self._policy,
            )
            embedder = self._embedder_factory()
            index = build_index(document, embedder)
            retriever = HybridRetriever(embedder, self._reranker_factory())
        except SparkDocsError as exc:
            raise WorkspaceError(str(exc)) from exc
        except Exception as exc:
            raise WorkspaceError("The document could not be indexed safely.") from exc
        self._document = document
        self._index = index
        self._retriever = retriever
        return self.status()

    def search(self, question: str) -> dict[str, object]:
        normalized = question.strip()
        if not normalized:
            raise WorkspaceError("Question must not be empty.")
        if len(normalized) > 2_000:
            raise WorkspaceError("Question exceeds the 2,000-character limit.")
        if self._index is None or self._retriever is None:
            raise WorkspaceError("Ingest a document before searching it.")
        try:
            evidence = self._retriever.retrieve(
                self._index,
                normalized,
                result_limit=6,
                max_characters=12_000,
            )
        except Exception as exc:
            raise WorkspaceError("The document search could not be completed safely.") from exc
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "document_sha256": self._index.document_sha256,
                "evidence": [],
                "limitations": [
                    "Retrieval can miss relevant passages; no answer should be inferred."
                ],
            }
        return {
            "status": "evidence_found",
            "document_sha256": self._index.document_sha256,
            "evidence": [
                {"locator": section.locator, "text": section.text} for section in evidence
            ],
            "limitations": [
                "Evidence retrieval does not prove that a proposed answer is correct or complete.",
                "Citations must use only the exact locators returned in this result.",
            ],
        }

    def status(self) -> WorkspaceStatus:
        if self._document is None or self._index is None:
            return WorkspaceStatus(loaded=False)
        return WorkspaceStatus(
            loaded=True,
            document_sha256=self._document.sha256,
            kind=self._document.kind.value,
            sections=len(self._document.sections),
            chunks=len(self._index.chunks),
            characters=self._document.character_count,
        )

    def clear(self) -> WorkspaceStatus:
        self._document = None
        self._index = None
        self._retriever = None
        return self.status()
