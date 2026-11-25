# Threat model

SparkDocs assumes documents, filenames, current and prior questions, and model output are hostile input. The local user and configured document root are the trust boundary; a document is never an authority source for tool use.

| Threat | Enforced control | Residual risk |
|---|---|---|
| Path traversal through MCP | Server startup fixes one existing root; tools accept relative paths and resolve them strictly below it | A user who configures an overly broad root grants broad read scope |
| Archive bombs and malformed OOXML | Member, expansion, ratio, section, and character limits before extraction | Parsers remain in-process and are not a malware sandbox |
| Prompt injection in documents | MCP returns evidence only; skill treats content as data; app provider receives no tools | A model can still misunderstand malicious prose |
| Evidence or secret leakage | No telemetry; cloud chat sends bounded evidence and limited prior questions after provider consent; MCP has no cloud provider/key tool | The selected provider sees submitted excerpts and question context; the MCP client sees returned evidence |
| Unsupported citations | Exact locator allowlist and skill abstention rules | Locator validity does not independently prove entailment |
| Resource exhaustion | Upload, extraction, question, candidate, result, and character bounds | First local model download and inference time vary by hardware |

The MCP transport is local stdio only. It does not authenticate network users or expose HTTP. Do not wrap it in a network listener without a new threat model and authentication design.
