"""SparkDocs application core."""

from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.models import Chunk, DocumentIndex, DocumentKind, IngestedDocument, Section
from sparkdocs.retrieval import HybridRetriever, build_index

__all__ = [
    "Chunk",
    "DocumentIndex",
    "DocumentKind",
    "HybridRetriever",
    "IngestedDocument",
    "Section",
    "UploadPolicy",
    "build_index",
    "ingest_upload",
]
