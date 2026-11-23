"""Safe product-level errors for upload, extraction, and Q&A failures."""


class SparkDocsError(Exception):
    """Base error safe to translate into a user-facing category."""


class UploadRejectedError(SparkDocsError):
    """The upload violates the declared policy."""


class ExtractionError(SparkDocsError):
    """The validated file could not be extracted safely."""


class ExtractionLimitError(ExtractionError):
    """Extraction exceeded a configured resource or content bound."""


class EmptyDocumentError(ExtractionError):
    """No supported text was found in the document."""


class QuestionError(SparkDocsError):
    """A question is invalid or has no matching evidence."""


class InsufficientEvidenceError(QuestionError):
    """Local retrieval found no evidence to send for this question."""


class ProviderRequestError(SparkDocsError):
    """The selected provider could not complete a generation request."""


class ProviderResponseLimitError(SparkDocsError):
    """A provider response exceeded the configured application limit."""
