# wllm

**vLLM for AI safety researchers: fast serving plus runtime-selectable model internals.**

wllm is an OpenAI-compatible inference server with a generic, versioned white-box extraction API over vLLM. Normal generation stays trace-free by default; extraction is enabled only through `/v1/extract` or `/v1/traces`.

## Status

Implemented:

- Flat `src/` layout with no `src/wllm/` package.
- `wllm` CLI with lazy vLLM imports.
- Non-streaming `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/completions`.
- Generic `POST /v1/extract`, `POST /v1/traces`, and `GET /v1/extraction-schema`.
- Versioned schemas: `wllm.extraction.v1` and `wllm.trace.v1`.
- Selector normalization for layers, positions, attention `previous_token`, and hidden-state pooling metadata.
- Request-scoped collector registry used by tests for cleanup and isolation semantics.
- Conservative resource limits and OpenAI-style error envelopes.
- `.npz` and conditional `.pt` artifact storage with SHA-256 manifests and path traversal protection.
- Optional research adapter protocol under `src/research/`; paper-specific terms are not part of the server API.

Capability-gated:

- Token IDs and bounded top-k generated-token logprobs are supported through vLLM generation outputs.
- Prompt logprobs are opt-in with `extract.logprobs.include_prompt=true` when the active vLLM `SamplingParams` supports `prompt_logprobs`.
- Raw logits are unsupported because the public vLLM generation output exposes logprobs, not logits.
- Exact entropy is unsupported unless a future runtime path exposes the complete distribution. Approximate entropy can be explicitly requested from renormalized top-k logprobs.
- Selected transformer-block token hidden states are conditionally supported for models whose vLLM configuration advertises the `pooling` runner and runs with `tensor_parallel_size=1`. wllm runs the completed token sequence through an isolated vLLM pooling runner with temporary scoped hooks and records the capture site as `transformer_block_output`.
- Tensor-parallel hidden-state capture remains unsupported until aggregation of sharded intermediate activations is implemented. The normal serving runner never receives hooks.
- Attention weights are reported unsupported when the active vLLM path or fused attention backend does not expose them.

Performance posture:

- Server orchestration, schemas, planning, tracing, and artifact handling are implemented in Python.
- The generation hot path stays inside vLLM. Normal OpenAI-compatible requests do not build extraction plans, collectors, or artifacts.
- Extraction postprocessing uses NumPy/PyTorch-compatible dense arrays for artifact data instead of doing Python work inside token-level generation loops.
- Native code is not introduced speculatively. For GPU-bound extraction kernels, Triton is preferred after profiling shows a real bottleneck; PyTorch C++/CUDA custom operators are reserved for cases where Triton and existing vLLM/PyTorch primitives are insufficient.
- Server components are not rewritten in C or Rust for speculative performance gains.

## Installation

For unit tests and API shape work:

```bash
pip install -e '.[test]'
```

For production vLLM serving:

```bash
pip install -e '.[vllm]'
```

Supported Python: 3.10 or newer.

Validated vLLM version: `0.10.2`. The production `vllm` extra installs `vllm==0.10.2`, and the runtime rejects other vLLM versions with a structured `503` instead of guessing private API paths.

## Layout

```text
src/
  cli.py
  server/
  schemas/
  runtime/
  extractors/
  tracing/
  artifacts/
  research/
tests/
  unit/
  integration/
```

## Start The Server

```bash
wllm serve Qwen/Qwen3-0.6B --local-files-only
```

Useful options:

```text
--host
--port
--dtype
--tensor-parallel-size
--gpu-memory-utilization
--max-model-len
--tokenizer
--served-model-name
--api-key
--seed
--trust-remote-code
--local-files-only
--artifact-dir
--log-level
```

`--served-model-name` changes the model name returned by `GET /v1/models` and in generation responses. If omitted, the loaded model name is used.

`--api-key` requires an `Authorization` header (Bearer token or bare token) on every request.

`--local-files-only` sets Hugging Face offline environment flags before vLLM initialization. If files are missing locally, server startup fails with a structured runtime error instead of silently downloading model files.

`--max-model-len`, `--tokenizer`, and `--seed` are passed to vLLM when supported by the active vLLM version. Unsupported sampling or constructor parameters are rejected with a structured error rather than silently ignored.

## Compatibility with `vllm serve`

wllm is a drop-in replacement for **common** vLLM OpenAI serving workflows, not a complete reimplementation of `vllm serve`. It intentionally accepts a focused subset of vLLM options and adds `--artifact-dir`, `--api-key`, and `--served-model-name`.

Supported common options: `--host`, `--port`, `--dtype`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--tokenizer`, `--served-model-name`, `--api-key`, `--seed`, `--trust-remote-code`, `--local-files-only`.

Known unsupported vLLM options include: `--quantization`, `--kv-cache-dtype`, `--device`, `--load-format`, `--config-format`, `--worker-use-ray`, `--pipeline-parallel-size`, `--enable-prefix-caching`, `--speculative-model`, `--num-scheduler-steps`, `--multi-step-stream-outputs`, `--allowed-local-media-path`, `--conversation-template`, `--enable-auto-tool-choice`, `--tool-call-parser`, `--chat-template`, `--chat-template-content-format`, `--enable-reasoning`, `--reasoning-parser`, `--lora-modules`, `--prompt-adapters`, `--max-logprobs`, `--disable-log-stats`, `--use-v2-block-manager`, `--enable-prompt-span`, `--enable-chunked-prefill`, `--embedding-mode`, `--swap-space`, `--max-num-batched-tokens`, `--max-num-seqs`, `--generation-config`, `--model-loader-extra-config`, `--ignore-patterns`, `--preemption-mode`, `--scheduler-delay-factor`, `--enable-torch-compile`, `--torchcompile-max-bs`, `--distributed-executor-backend`, `--max-local-adapters`, `--model-impl`, `--enable-async-output-proc`, `--enable-p2p-check`, `--enable-kv-transfer`, `--kv-transfer-config`, and options related to disaggregated serving, multimodal inputs, tool calling, structured outputs, and quantization.

Unsupported options are rejected by the CLI parser or, if they map to vLLM constructor/sampling parameters, by runtime validation with a structured error.

## OpenAI-Compatible Requests

Chat:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

Text completion:

```bash
curl http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Say hello:","max_tokens":16}'
```

Streaming is not implemented in this initial server and is rejected explicitly when `stream=true`. Sampling fields are translated to vLLM `SamplingParams`; explicitly requested sampling fields that the active vLLM version does not support are rejected with a structured `422` error instead of being silently ignored.

## Extraction Requests

Bounded inline token and logprob extraction:

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "messages":[{"role":"user","content":"Explain refusal behavior briefly."}],
    "max_tokens":64,
    "extract":{"tokens":true,"logprobs":{"top_k":5,"include_prompt":true}}
  }'
```

Persisted trace request:

```bash
curl http://localhost:8000/v1/traces \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "prompt":"Summarize uncertainty:",
    "max_tokens":32,
    "extract":{"tokens":true,"logprobs":{"top_k":5},"artifacts":{"format":"npz","include":["logprobs"]}}
  }'
```

`/v1/extract` returns generation plus bounded extraction results inline when available. `/v1/traces` uses the same request schema, writes a normalized JSON trace bundle, and returns a `trace_manifest` with the bundle path, byte size, SHA-256 digest, creation time, and trace ID. Tensor artifacts requested through `extract.artifacts` are written separately.

`GET /v1/extraction-schema` returns `wllm.extraction.v1`, the live Pydantic JSON schema for extraction requests, selector semantics, configured resource limits, and runtime capabilities.
When vLLM exposes or configures an attention backend, capability metadata includes `attention_backend` and echoes it in the attention capability details.
Hidden-state capability metadata reports whether the loaded model advertises a vLLM `pooling` runner and whether the worker configuration is single-rank. Without that runner, with tensor parallelism enabled, with `gpu_memory_utilization > 0.5`, or without runtime `hidden_size` metadata for capture-size enforcement, hidden-state extraction returns `hidden_states_unavailable`.

## Trace Schema

Trace responses use `wllm.trace.v1` and include:

- `generation`: OpenAI-style generation summary.
- `trace.tokens`: token IDs and decoded tokens when requested.
- `trace.spans`: prompt and generated token spans over the final token sequence.
- `trace.logprobs`: per-token selected `token_id`/`token`/`logprob` fields and top-k alternatives for generated tokens, plus prompt-token rows when `include_prompt=true`.
- `trace.hidden_states` and `trace.attentions`: tensor records when supported by the active runtime.
- `trace_manifest`: persisted JSON trace-bundle manifest for `/v1/traces`.
- `artifacts`: artifact manifests with byte size, SHA-256, tensor names, shapes, capture dtypes, storage dtypes, and trace ID.
- `metadata`: sampling, resolved selectors, capabilities, and timing fields.

When model topology is available from vLLM, layer and head selectors are validated against it before extraction proceeds. Trace metadata records resolved selectors for tensor requests, including normalized layer indexes, token positions, and pooling metadata.

## Selectors

Layer selectors support integers, integer lists, negative indexes, `all`, `middle`, and `middle_third`.

Position selectors support integers, integer lists, negative indexes, `prompt`, `generated`, `last`, and `last_generated` over the final token sequence after chat-template rendering.

Attention key positions additionally support `previous_token`, which maps each valid query position `q` to key `q - 1`.

Hidden-state pooling metadata supports `null`, `mean`, `max`, and `last`.

In the current vLLM runtime, hidden-state tensor values are captured from selected transformer-block modules on an isolated pooling runner. This is post-generation hidden-state replay over the exact prompt token IDs plus generated token IDs reported by vLLM, not online capture from the original autoregressive generation forwards. This path is opt-in per extraction request, guarded by a per-runner lock, and currently limited to `tensor_parallel_size=1`. For validated vLLM 0.10.2, the isolated pooling runner is initialized with eager execution and in-process V1 engine mode so scoped module hooks observe real forward outputs; the normal generation runner keeps its standard vLLM execution path.

## Resource Limits

Defaults are conservative:

- `max_top_k`: 50
- `max_selected_layers`: 8
- `max_selected_heads`: 32
- `max_selected_positions`: 256
- `max_inline_tensor_bytes`: 1 MB
- `max_total_captured_tensor_bytes`: 64 MB
- `max_artifact_bytes`: 256 MB
- `large_extraction_enabled`: false

Requests above configured limits return `413` with an OpenAI-style error envelope.

Inline token-id and logprob numeric payloads are checked against `max_inline_tensor_bytes`. Reduce `max_tokens` or `extract.logprobs.top_k` when inline extraction exceeds this limit.

Large tensor requests require three things: an artifact request that includes the tensor family, `extract.artifacts.allow_large=true`, and server-side `large_extraction_enabled=true`. Full hidden-state or attention dumps are rejected unless they are artifact-backed and pass the hard byte limits. Hidden-state capture is also checked against `max_total_captured_tensor_bytes` before the pooling runner starts, using the raw captured shape `requested_layers * total_tokens * hidden_size * dtype_bytes`, because the runtime captures the full completed token sequence before applying requested position selectors. The default server configuration keeps large extraction disabled.

Logprob artifacts require `extract.logprobs`; artifact inclusion alone does not silently request extra vLLM outputs.

When a captured tensor dtype cannot be represented by the selected storage format, wllm converts explicitly and records both the original capture dtype and the stored dtype. For example, bfloat16 hidden states stored as JSON or NPZ are converted to float32 values with `capture_dtype="torch.bfloat16"` and `storage_dtype="float32"` metadata.

## Capability Errors

Unsupported extraction returns:

```json
{
  "error": {
    "message": "Selected attention weights are not exposed by the active vLLM serving path.",
    "type": "unsupported_extraction",
    "param": "extract.attentions",
    "code": "attention_weights_unavailable",
    "details": {}
  }
}
```

The runtime never returns placeholder tensors or synthetic internals.

If the active vLLM `SamplingParams` does not expose `logprobs` or `prompt_logprobs`, extraction requests for those outputs return `unsupported_extraction` with `token_logprobs_unavailable` or `prompt_logprobs_unavailable`. Ordinary OpenAI sampling fields that the active vLLM version does not support remain `422` validation errors.

If the model does not support vLLM's pooling runner, or if tensor parallelism is enabled, hidden-state requests return `unsupported_extraction` with `hidden_states_unavailable` and capability details explaining the unavailable condition.

## vLLM Compatibility Notes

The standard generation path uses vLLM `LLM.generate` and `SamplingParams`. Hidden-state extraction uses isolated compatibility code in `runtime/vllm_compat.py`: vLLM `LLM.encode`, the pooling model conversion path, and scoped `apply_model`/model-executor access for temporary hooks. This private surface is version-guarded to vLLM 0.10.2 and tested with vLLM 0.10.2 in the reference integration environment.

## Research Adapters

Research adapters consume generic trace objects. A `/v1/traces` response includes a `trace_manifest` that can be loaded from the configured artifact directory:

```python
from artifacts import load_artifact
from research.token_baselines import TokenBaselineAdapter
from tracing.serialization import load_trace_bundle

trace = load_trace_bundle("./wllm-artifacts", trace_response["trace_manifest"])
result = TokenBaselineAdapter().run(trace)
tensors = [load_artifact("./wllm-artifacts", manifest) for manifest in trace.artifacts]
```

Custom adapters should depend on `TraceEnvelope`, artifact manifests, and tensor files rather than server routes or paper-specific request fields:

```python
from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class GeneratedTokenCount:
    name = "generated_token_count"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del options
        start, end = trace.trace.spans["generated"]
        return ResearchResult(
            name=self.name,
            status="ok",
            values={"count": end - start},
        )
```

RAUQ, EigenScore, probes, and ActMap belong under `src/research/` and must consume trace tensors or artifacts. They do not define public request fields or server routes.

## Tests

Unit tests do not require vLLM, a GPU, or model downloads:

```bash
pytest tests/unit -q
```

Integration tests are gated:

```bash
WLLM_TEST_MODEL=/path/to/local/model pytest -m integration
```

The integration suite is skipped unless vLLM is installed and `WLLM_TEST_MODEL` points to a local model. It includes a token/logprob smoke test and a hidden-state smoke test using a selected middle layer. The validated reference run used vLLM 0.10.2, PyTorch 2.8 CUDA wheels, Qwen/Qwen3-0.6B from a local Hugging Face cache, `WLLM_TEST_MAX_MODEL_LEN=1024`, and `WLLM_TEST_GPU_MEMORY_UTILIZATION=0.35`. Use a reference model whose vLLM configuration supports the `pooling` runner and `tensor_parallel_size=1` for the hidden-state test. Attention coverage remains capability-gated until the active backend exposes weights.
