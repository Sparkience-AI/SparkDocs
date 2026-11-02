# Privacy and data handling

SparkDocs is local-first, but its current Q&A provider is Google Gemini.

## Local application

- Uploaded bytes are validated and extracted in the Streamlit process memory.
- The application does not intentionally write uploads, extracted text, questions, answers, or API
  keys to disk.
- The document and API key remain in Streamlit session state until the user clears the session, the
  process restarts, or the hosting platform expires the session.
- Selecting **Clear document and API key** removes the application-owned session values. Python and
  hosting-platform memory reclamation is not a cryptographic erasure guarantee.

## Provider transmission

When the user asks a question, SparkDocs sends the question, the application-owned grounded-answer
instructions, and up to six lexically selected document sections to Google. Google processing and
retention are governed by the user's account and Google's applicable terms.

The API key is passed directly to a session-owned client. SparkDocs does not copy it into environment
variables, URLs, logs, caches, or files.

## Hosted demo

A hosted Streamlit server necessarily receives the upload and key in server memory. Use a restricted,
low-quota, disposable key and non-sensitive test documents. SparkDocs alpha is not approved for
confidential, personal, regulated, legal, medical, or financial documents.

SparkDocs has no telemetry pipeline. Operators must not add prompt or content logging without a new
privacy review and explicit user-facing disclosure.
