# Privacy and data handling

SparkDocs is local-first. General and document answers use the explicitly selected Google
Gemini, OpenAI, or Anthropic Claude cloud provider.

## Local application

- Uploaded bytes are validated and extracted in the Streamlit process memory.
- The application does not intentionally write uploads, extracted text, questions, answers, or API
  keys to disk.
- The current document and its chat history remain in Streamlit session state. Replacing or removing
  the document resets the conversation. Active provider settings and keys are held in session
  memory and cleared when the session ends or the user clears it.
- Selecting **Clear document, provider settings, and keys** removes the application-owned session
  values. Python and hosting-platform memory reclamation is not a cryptographic erasure guarantee.

## Provider transmission

Only after the user checks the provider-specific transmission confirmation and asks a question,
SparkDocs can make a cloud request. A document answer sends the current question, the
application-owned grounded-answer instructions, and a character-bounded set of freshly retrieved
excerpts to the selected provider. A follow-up can also
include up to two recent user questions from successful turns with that same provider, with a
total context limit of 600 characters. Previous assistant answers, local-only questions, and
excerpts from earlier turns are excluded from the document-answer context. In automatic mode with
a document loaded, the provider may first classify the current question using bounded prior
document questions. The classification request includes no document excerpts. General answers
include no document excerpts, and can include up to two prior general exchanges with the same
provider, capped at 400 question characters and 600 answer characters each. Local evidence lookup
makes no provider request.
Provider processing and retention are governed by the user's account, plan, configuration, and the
provider's applicable terms. OpenAI requests set `store=false`; this application setting does not
replace the provider's account-level data controls or policies.

Dense embedding and cross-encoder reranking run locally through FastEmbed. Their model weights are
downloaded from Hugging Face on first use and cached by FastEmbed. Document text is processed by
those local models and is not sent to Hugging Face for inference. The model cache can persist after
the Streamlit session ends, but SparkDocs does not intentionally put document content in that cache.

Each API key is passed directly to a session-owned client. SparkDocs does not copy it into
environment variables, URLs, logs, caches, or files.

## Hosted demo

A hosted Streamlit server necessarily receives the upload and key in server memory. Use a restricted,
low-quota, disposable key and non-sensitive test documents. SparkDocs alpha is not approved for
confidential, personal, regulated, legal, medical, or financial documents.

SparkDocs has no telemetry pipeline. Operators must not add prompt or content logging without a new
privacy review and explicit user-facing disclosure.

## Local MCP

The stdio MCP process reads only a user-configured root and keeps one document/index in memory.
It returns bounded evidence text to the connected local MCP client. It has no provider-key argument
and makes no model request. `clear_document` drops application-owned references; operating-system
memory reclamation is not cryptographic erasure.
