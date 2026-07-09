# Changelog

## Unreleased

### Researcher Ergonomics (white-box UQ / interpretability)
- Chosen token log probabilities are now first-class in extraction traces
  (`trace.tokens.chosen_logprobs`). Extraction paths always capture at least
  the selected token's logprob (cheap `logprobs=1` from backend) so common
  UQ baselines (min prob, sequence prob, length-normalized, etc.) are trivial
  even when only `"tokens": true` is requested.
- Added generic, method-agnostic helpers in `research.features`:
  `chosen_logprobs(trace)`, `hidden_states_matrix(...)`, `last_token_hidden(...)`.
  These turn traces + artifacts into ready-to-use numpy arrays for probing,
  spectral methods, token-prob UQ, etc.
- Prompt-only / pre-generation hidden state extraction is supported:
  use `max_tokens: 0` + hidden state selectors on `"prompt"` or `"last"`
  positions (or `"last_generated"` when generation occurs). Useful for
  statement-level truth probes before answering.
- `benchmarking/` directory added with analysis of top white-box UQ and
  hallucination detection methods (token-prob baselines, EigenScore/INSIDE,
  RAUQ, SAPLMA-style probes, Semantic Entropy). Contains data contracts,
  current wllm mappings, friction points, and synthesized general (non
  method-specific) recommendations.
- Updated documentation (README, dataset workflow comments) and research
  surface to highlight the new helpers and benchmarking evidence.

## 0.1.0 - 2026-06-27

First usable release of `wllm`: OpenAI-compatible vLLM serving with explicit,
opt-in white-box extraction for safety research workflows.

### Supported

- Non-streaming OpenAI-compatible endpoints:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- Research extraction endpoints:
  - `POST /v1/extract` for inline trace responses
  - `POST /v1/traces` for persisted trace bundles and tensor artifacts
  - `GET /v1/extraction-schema` for request schema, selectors, limits, and
    runtime capabilities
- Token tracing with prompt and generated token IDs.
- Decoded token strings when token or logprob extraction is requested.
- Generated top-k logprobs and prompt logprobs when exposed by vLLM.
- Approximate entropy over renormalized top-k logprobs when explicitly allowed.
- Selected hidden-state capture:
  - replay mode through an isolated vLLM pooling runner
  - online mode when started with `--enable-online-hidden-states`
- Selector resolution for layers, heads, and token positions, including
  `middle`, `middle_third`, `prompt`, `generated`, `last`, and
  `last_generated`.
- NPZ tensor artifacts with optional compression.
- PT tensor artifacts when PyTorch is installed.
- Safe trace and artifact loaders that validate path containment, byte size,
  SHA-256 digest, tensor names, shapes, and dtypes.
- Server-side resource limits for top-k, selected layers/heads/positions,
  inline tensor bytes, captured tensor bytes, and artifact bytes.
- Serialized access to the shared active vLLM generation runner so concurrent
  HTTP requests cannot overlap scheduler state or online hidden-state hooks.
- OpenAI-style structured error envelopes for validation errors, resource
  limits, unsupported extraction, runtime unavailability, authentication
  failure, unknown endpoints, wrong HTTP methods, and unexpected server errors.
  Envelopes include message, type, status, param, code, and details fields.
- Static API-key middleware for single-user or trusted-network deployments.
- `wllm doctor` environment preflight checks for Python, installed package
  metadata, exact vLLM version, optional PyTorch/Transformers packages, and
  local path-like model arguments.
- Dataset workflow example for JSONL prompts, persisted traces, artifact
  loading, and a minimal token-baseline adapter.
- Curated release evidence in `reports/release-v0.1-readiness.md` and the
  multi-architecture validation report/results.

### Experimental

- Replay attention extraction behind `--enable-attention-weights`.
  - This path uses a separate Transformers/PyTorch model with
    `output_attentions=True` and `use_cache=False`.
  - It is intended for bounded research traces, not low-latency serving.
  - Attention tensors scale quadratically with sequence length.
  - Online attention capture from the active vLLM generation path is not
    supported.

### Validated Assumptions

- Python: 3.10+
- vLLM: exactly `0.10.2`
- Tested platform: Linux
- Hidden-state replay requires `tensor_parallel_size=1`.
- Hidden-state replay usually requires `gpu_memory_utilization <= 0.5` because
  it initializes a second model runner.
- Integration tests require a local model path or cached model configured with
  `WLLM_TEST_MODEL`.

### Not Supported

- Streaming responses.
- Full OpenAI API parity.
- LoRA, quantization flags, speculative decoding, multimodal inputs, tool
  calling, structured outputs, reasoning parsers, custom chat templates, and
  pipeline parallelism.
- Raw logits from vLLM generation output.
- Exact entropy over the full token distribution.
- Online attention capture from the active vLLM generation runner.
- Multi-GPU hidden-state extraction.
- Paper-specific RAUQ, EigenScore, ActMap, or probe methods as server request
  fields.

### Known Operational Notes

- Normal chat and completion requests are trace-free: they do not allocate
  collectors, install hooks, load replay models, activate trace context, or
  write artifacts.
- Artifact writes are synchronous on the request path.
- Optional replay paths may load additional model instances and require
  substantially more VRAM than trace-free generation.
- `RuntimeCapabilities` and `/v1/extraction-schema` should be checked by client
  scripts before sending hidden-state or attention requests.
