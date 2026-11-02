"""SparkDocs application core."""

from sparkdocs.ingestion import UploadPolicy, ingest_upload
from sparkdocs.models import DocumentKind, IngestedDocument, Section

__all__ = [
    "DocumentKind",
    "IngestedDocument",
    "Section",
    "UploadPolicy",
    "ingest_upload",
]
