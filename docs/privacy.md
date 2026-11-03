# Privacy and data handling

SparkDocs is local-first, but grounded answer generation uses the explicitly selected Google
Gemini, OpenAI, or Anthropic Claude cloud provider.

## Local application

- Uploaded bytes are validated and extracted in the Streamlit process memory.
- The application does not intentionally write uploads, extracted text, questions, answers, or API
  keys to disk.
- The document, provider selections, and entered API keys remain in Streamlit session state until
  the user clears the session, the process restarts, or the hosting platform expires the session.
- Selecting **Clear document, provider settings, and keys** removes the application-owned session
  values. Python and hosting-platform memory reclamation is not a cryptographic erasure guarantee.

## Provider transmission

Only after the user checks the provider-specific transmission confirmation and asks a question,
SparkDocs sends the question, the application-owned grounded-answer instructions, and a
character-bounded set of locally retrieved excerpts to the selected provider.
Provider processing and retention are governed by the user's account, plan, configuration, and the
provider's applicable terms. OpenAI requests set `store=false`; this application setting does not
replace the provider's account-level data controls or policies.

Dense embedding and cross-encoder reranking run locally through FastEmbed. Their model weights are
downloaded from Hugging Face on first use and cached by FastEmbed. Document text is processed by
those local models and is not sent to Hugging Face for inference. The model cache can persist after
the Streamlit session ends, but SparkDocs does not intentionally put document content in that cache.

Each API key is passed directly to a session-owned client. SparkDocs does not copy it into
environment variables, URLs, logs, caches, or files. Provider-specific keys and model selections
remain in Streamlit session memory until cleared or the session expires.

## Hosted demo

A hosted Streamlit server necessarily receives the upload and key in server memory. Use a restricted,
low-quota, disposable key and non-sensitive test documents. SparkDocs alpha is not approved for
confidential, personal, regulated, legal, medical, or financial documents.

SparkDocs has no telemetry pipeline. Operators must not add prompt or content logging without a new
privacy review and explicit user-facing disclosure.
