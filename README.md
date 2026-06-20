# wllm

**vLLM for AI safety researchers: fast serving plus runtime-selectable model internals.**

wllm is an OpenAI-compatible inference server with a generic, versioned white-box extraction API over vLLM. Normal generation stays trace-free by default; extraction is enabled only through `/v1/extract` or `/v1/traces`.

## Release Status

**Version 0.1.0** — initial release.

- **353 unit tests pass** (no GPU/vLLM required).
- **4 integration tests** (smoke: token/logprob, replay HS, online HS, repeated replay) require GPU + vLLM + local model.
- **3 model architectures validated**: Qwen3, Llama, Mistral (see Capability Matrix below).
- **1 benchmarked configuration**: Qwen3-0.6B on RTX 5090 (24 GB), vLLM 0.10.2.

This release is suitable for researchers who want to prototype extraction workflows on small-to-medium models (up to ~6B parameters for hidden-state replay). Production deployments and streaming are not supported. See Known Limitations.

## What wllm Is

- An **OpenAI-compatible inference server** exposing `/v1/models`, `/v1/chat/completions`, and `/v1/completions`.
- A **white-box extraction API** (`/v1/extract`, `/v1/traces`) for token IDs, logprobs, and hidden states with versioned schemas.
- A **drop-in replacement** for common non-streaming `vllm serve MODEL` workflows, accepting a focused subset of vLLM CLI options.
- **Extraction-free by default**: normal generation creates no collectors, hooks, or artifacts.

## What wllm Is Not

- **Not a streaming server.** `stream=true` is explicitly rejected on all endpoints.
- **Not a complete vLLM reimplementation.** Unsupported vLLM options (quantization, LoRA, speculative decoding, multimodal, etc.) are rejected.
- **Not a model training or fine-tuning framework.**
- **Not a replacement for vLLM's full feature set.** It targets AI safety research workflows that need runtime model internals alongside standard serving.

## Installation

Requires Python 3.10 or newer.

For unit tests and API shape work (no GPU/vLLM needed):

```bash
pip install -e '.[test]'
```

For production vLLM serving:

```bash
pip install -e '.[vllm]'
```

**Supported vLLM version:** `0.10.2` (exact match enforced at runtime). The production `vllm` extra installs `vllm==0.10.2`. The runtime rejects other vLLM versions with a structured HTTP 503 error.

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

## Start the Server

```bash
wllm serve Qwen/Qwen3-0.6B --local-files-only
```

Accepts both HuggingFace IDs and local filesystem paths. Full option list:

```text
--host                  (default: 127.0.0.1)
--port                  (default: 8000)
--dtype                 (default: auto)
--tensor-parallel-size  (default: 1)
--gpu-memory-utilization (default: 0.9)
--max-model-len
--tokenizer
--served-model-name
--api-key
--seed
--trust-remote-code
--local-files-only
--prewarm-hidden-states
--enable-online-hidden-states
--artifact-dir          (default: ./wllm-artifacts)
--log-level             (default: info, choices: debug/info/warning/error)
```

Key options:

- `--served-model-name`: changes the model name in `/v1/models` and generation responses.
- `--api-key`: requires an `Authorization` header (Bearer token or bare token) on every request.
- `--local-files-only`: sets `HF_HUB_OFFLINE=1` and related env vars before vLLM init. Fails with a structured error if model files are missing, rather than downloading.
- `--prewarm-hidden-states`: initializes the isolated pooling runner at startup instead of on the first hidden-state request (~3.3s cold start avoided).
- `--enable-online-hidden-states`: enables `capture_mode="online"` for hidden states. Starts the generation runner in eager in-process mode, which changes generation performance.
- `--artifact-dir`: directory for persisted traces and NPZ/PT artifacts.

## Standard OpenAI Client Usage

wllm is compatible with the standard `openai` Python library. Point the client at your wllm server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # or your --api-key value
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=16,
)
print(response.choices[0].message.content)

# Text completion
response = client.completions.create(
    model="Qwen/Qwen3-0.6B",
    prompt="Say hello:",
    max_tokens=16,
)
print(response.choices[0].text)

# List models
models = client.models.list()
for m in models.data:
    print(m.id)
```

## Non-Streaming Limitation

**wllm 0.1.0 does not implement streaming.** Setting `stream=true` on any endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/extract`, `/v1/traces`) returns HTTP 422 with:

```json
{"error": {"message": "Streaming ... is not implemented.", "code": "streaming_not_implemented", "type": "invalid_request_error", "param": "stream"}}
```

## Normal Generation (curl)

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

Normal generation requests go directly through vLLM without building extraction plans, collectors, hooks, or artifacts. Sampling fields (`temperature`, `top_p`, `top_k`, `stop`, `n`, `seed`, `presence_penalty`, `frequency_penalty`, `logprobs`) are translated to vLLM `SamplingParams`. Unsupported sampling fields are rejected with a structured 422 error.

## Token & Logprob Extraction

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

This returns a `wllm.trace.v1` response with:
- `trace.tokens.token_ids` — exact integer token IDs from vLLM generation.
- `trace.tokens.tokens` — decoded token strings.
- `trace.logprobs.generated` — per-token logprobs with top-k alternatives.
- `trace.logprobs.prompt` — prompt-token logprob rows (when `include_prompt=true`).

Token/logprob extraction is verified by integration tests and 353 unit tests. Approximate entropy is available via `extract.logprobs.entropy=true` with `allow_approximate_entropy=true`, computed from renormalized top-k logprobs.

## Hidden-State Replay & Extraction

Replay mode (default) captures hidden states after generation using an isolated vLLM pooling runner:

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "prompt":"Explain how transformer blocks process information.",
    "max_tokens":32,
    "extract":{
      "hidden_states":[
        {"layers":"middle_third","positions":"generated","pool":"mean"}
      ]
    }
  }'
```

Requirements: `tensor_parallel_size=1`, `gpu_memory_utilization <= 0.5`.

Online capture mode (opt-in via `--enable-online-hidden-states`) captures during generation:

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "prompt":"Explain how transformer blocks process information.",
    "max_tokens":32,
    "extract":{
      "hidden_states":[
        {"layers":"middle","positions":"prompt","capture_mode":"online"}
      ]
    }
  }'
```

**Important**: Online hidden states are not numerically interchangeable with replay-mode hidden states (benchmarked relative L2 error up to 1.26, max absolute divergence 13.56 in tested cases). Use the appropriate capture mode for your research question. Online mode uses predictor/source position p-1 for generated-token selectors.

## Artifact Export

Request NPZ artifact output alongside the trace:

```bash
curl http://localhost:8000/v1/traces \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "prompt":"Summarize uncertainty:",
    "max_tokens":32,
    "extract":{
      "tokens":true,
      "logprobs":{"top_k":5},
      "hidden_states":[{"layers":"middle_third","positions":"generated"}],
      "artifacts":{"format":"npz","include":["logprobs","hidden_states"]}
    }
  }'
```

NPZ is compressed by default. For lower serialization latency at the cost of larger files, use uncompressed:

```json
"artifacts":{"format":"npz","compression":"uncompressed","include":["hidden_states"]}
```

PT format is also supported:

```json
"artifacts":{"format":"pt","include":["hidden_states"]}
```

All artifacts include SHA-256 digests, byte sizes, tensor shapes/dtypes, and capture/storage dtype metadata in their manifests. bf16 tensors are converted to float32 for NPZ with `capture_dtype="torch.bfloat16"` and `storage_dtype="float32"` recorded.

## Artifact Loading

Load artifacts from disk using the manifest returned in `/v1/traces` responses:

```python
from artifacts import load_artifact

# Load a single artifact by its manifest
tensors = load_artifact("./wllm-artifacts", trace_response["artifacts"][0])

# Or load by manifest dict
tensors = load_artifact("./wllm-artifacts", {
    "path": "traces/abc123/hidden_states_000.npz",
    "sha256": "a1b2c3...",
    "format": "npz",
})
```

`load_artifact` validates SHA-256 digest integrity and rejects path traversal attempts. It accepts both `ArtifactManifest` model instances and plain dicts.

## Trace-to-Artifact Mapping

When a trace is persisted via `/v1/traces`, the response connects trace metadata to on-disk artifact files:

```
/v1/traces response
├── trace_manifest          ──→  traces/<trace_id>/trace_bundle.json
│   ├── trace_id                              (full TraceEnvelope JSON)
│   ├── byte_size / sha256                    (validated on load)
│   └── schema_version                        (must equal "wllm.trace.v1")
│
└── artifacts[]             ──→  traces/<trace_id>/hidden_states_000.npz
    ├── path                                 (e.g., traces/<trace_id>/logprobs_000.npz)
    ├── format / byte_size / sha256           (validated on load)
    ├── tensor_shapes / tensor_dtypes          (per-tensor metadata)
    └── included_tensor_names                  (which extraction data is included)
```

Key relationships:
- The `trace_manifest` is a `TraceBundleManifest` that references the persisted JSON file. Use `load_trace_bundle(root, trace_manifest)` to load the full `TraceEnvelope`.
- Each entry in `artifacts` is an `ArtifactManifest` that references an on-disk NPZ or PT file. Use `load_artifact(root, artifact_manifest)` to load tensor data.
- Both manifests carry SHA-256 digests and byte sizes that are verified on every load.
- The trace ID ties artifacts back to their originating request: artifact paths are under `traces/<trace_id>/`.

The `/v1/extract` endpoint (non-persisting) returns the same `TraceEnvelope` shape inline, with `trace_manifest` null/absent and `artifacts` empty.

### One-liner loaders

```python
# Load a trace bundle from its manifest (one call)
trace = load_trace_bundle("./wllm-artifacts", response["trace_manifest"])

# Load an artifact from its manifest (one call)
tensors = load_artifact("./wllm-artifacts", response["artifacts"][0])

# Load all artifacts for a trace
all_tensors = {a["path"]: load_artifact("./wllm-artifacts", a) for a in response["artifacts"]}

# Load a trace bundle from a plain dict manifest
trace = load_trace_bundle("./wllm-artifacts", {"path": "traces/abc/trace_bundle.json", "sha256": "...", "byte_size": 1234, "schema_version": "wllm.trace.v1", "trace_id": "abc"})
```

## Dataset-Building Workflow

A typical researcher workflow: prompt dataset → extract traces → load artifacts → run analysis.

See `scripts/dataset_workflow.py` for a full runnable example with CLI flags and error handling.

```python
import json
import httpx
from artifacts import load_artifact
from tracing.serialization import load_trace_bundle
from research.token_baselines import TokenBaselineAdapter

client = httpx.Client(base_url="http://localhost:8000/v1", timeout=60)
ARTIFACT_DIR = "./wllm-artifacts"
MODEL = "Qwen/Qwen3-0.6B"

# 1. Read prompts from a JSONL dataset
with open("prompts.jsonl") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

# 2. Extract traces for each prompt
for i, prompt in enumerate(prompts):
    resp = client.post(
        "/traces",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 64,
            "extract": {
                "tokens": True,
                "logprobs": {"top_k": 5, "include_prompt": True},
                "hidden_states": [{"layers": "middle", "positions": "last_generated"}],
                "artifacts": {"format": "npz", "include": ["logprobs", "hidden_states"]},
            },
        },
    )
    resp.raise_for_status()
    trace_resp = resp.json()

    # 3. Load the persisted trace bundle (one-liner)
    trace = load_trace_bundle(ARTIFACT_DIR, trace_resp["trace_manifest"])

    # 4. Load artifact tensors (one-liners)
    tensors = {
        manifest["path"]: load_artifact(ARTIFACT_DIR, manifest)
        for manifest in trace_resp.get("artifacts", [])
    }

    # 5. Run a research adapter
    result = TokenBaselineAdapter().run(trace)
    print(f"[{i}] {prompt[:40]}... → {result.values}")

client.close()
```

All adapters (TokenBaseline, RAUQ, EigenScore, ActMap) consume the generic `TraceEnvelope` and artifact tensors. They define no server routes or paper-specific request fields.

## Trace Schema

Trace responses use `wllm.trace.v1` and include:

- `generation`: OpenAI-style generation summary with choices and usage.
- `trace.tokens`: token IDs and decoded tokens when requested.
- `trace.spans`: prompt and generated token spans over the final token sequence.
- `trace.logprobs`: per-token selected `token_id`/`token`/`logprob` fields and top-k alternatives.
- `trace.hidden_states`: tensor records when supported by the active runtime.
- `trace_manifest`: persisted JSON trace-bundle manifest (for `/v1/traces`).
- `artifacts`: artifact manifests with byte size, SHA-256, tensor names, shapes, dtypes.
- `metadata`: sampling params, resolved selectors, capabilities, and timing.

## Selectors

Layer selectors: integers, integer lists, negative indexes, `all`, `middle`, `middle_third`.

Position selectors: integers, integer lists, negative indexes, `prompt`, `generated`, `last`, `last_generated`.

Attention key positions: `previous_token` maps query position `q` to key `q-1`.

Hidden-state pooling: `null` (per-position), `mean`, `max`, `last`.

## Resource Limits

Server-side defaults (configurable at startup):

| Limit | Default |
|-------|---------|
| `max_top_k` | 50 |
| `max_selected_layers` | 8 |
| `max_selected_heads` | 32 |
| `max_selected_positions` | 256 |
| `max_inline_tensor_bytes` | 1 MB |
| `max_total_captured_tensor_bytes` | 64 MB |
| `max_artifact_bytes` | 256 MB |
| `large_extraction_enabled` | false |

Requests exceeding limits return HTTP 413 with an OpenAI-style error envelope.

## Capability Matrix

Tested on RTX 5090 (24 GB), vLLM 0.10.2, PyTorch 2.8.0+cu128, max_model_len=1024.

| Capability | Qwen3-0.6B | Llama-3.2-3B | Mistral-7B-Instruct-v0.3 |
|---|---:|---:|---:|
| Normal generation | ✅ | ✅ | ✅ |
| Chat completions | ✅ | ⚠️ base model, no template | ✅ |
| Text completions | ✅ | ✅ | ✅ |
| Token ID extraction | ✅ | ✅ | ✅ |
| Logprob extraction (top-k) | ✅ | ✅ | ✅ |
| Prompt logprobs | ✅ | ✅ | ✅ |
| Hidden state replay | ✅ | ✅ | ❌ VRAM |
| Hidden state online | ✅ | ❌ position mapping | ❌ VRAM |
| Prewarmed replay | ❌ tokenizer compat | ❌ tokenizer compat | ❌ VRAM |
| NPZ artifacts | ✅ | ✅ | ✅ (token/logprob only) |
| PT artifacts | ✅ | ✅ | ✅ (token/logprob only) |
| Artifact round-trip | ✅ | ✅ | ✅ |

Legend: ✅ = Working, ❌ = Failing/Unsupported, ⚠️ = Functional with caveat

**Architecture-specific notes:**

- **Llama-3.2-3B** is a base model with no `chat_template` in its tokenizer config. Chat completions fail with `ValueError`. Use `/v1/completions` with raw prompts instead.
- **Mistral-7B** (~14.5 GB weights) cannot fit a second model instance for replay hidden-state extraction on a 24 GB GPU. Online capture would require fixing the VRAM constraint first. Token/logprob extraction works normally.
- **Prewarmed replay** fails on all architectures due to a `transformers` 5.12.1 compatibility issue with vLLM 0.10.2 (missing `all_special_tokens_extended` attribute). Single-request replay extraction is unaffected; only repeated prewarmed calls trigger this path. See Known Limitations.

## Tested Model Architectures

| Model | Family | Parameters | Weight Size | HF Architecture |
|-------|--------|-----------|-------------|-----------------|
| Qwen3-0.6B | Qwen | 0.6B | ~1.1 GB | Qwen3ForCausalLM |
| Llama-3.2-3B | Llama | 3B | ~6.0 GB | LlamaForCausalLM |
| Mistral-7B-Instruct-v0.3 | Mistral | 7B | ~14.5 GB | MistralForCausalLM |

All three use Flash Attention backend on the V1 engine.

Additional models available locally but not yet integration-tested: Qwen3-4B, Qwen3-8B, Llama-3.1-8B.

## Performance Summary

Benchmarked on Qwen3-0.6B, RTX 5090 (24 GB), vLLM 0.10.2, max_model_len=1024, gpu_memory_utilization=0.35, batch size 1, 163 prompt + 32 generated tokens, warm runs (excluding first warmup and vLLM init).

| Mode | Median wall (ms) | vs raw vLLM |
|------|:---:|:---:|
| Raw vLLM generate | 90.9 | 1.00× |
| wllm completion (trace-free) | 90.0 | 0.99× |
| wllm extract tokens | 90.3 | 0.99× |
| wllm extract logprobs (top_k=5) | 102.3 | 1.13× |
| wllm trace + NPZ artifact | 103.8 | 1.14× |
| wllm hidden inline (1 layer, 1 pos) | 101.8 | 1.12× |
| wllm hidden NPZ (middle third) | 132.2 | 1.45× |
| wllm hidden NPZ uncompressed | 128.3 | 1.41× |
| wllm hidden PT | 128.2 | 1.41× |

Key takeaways:
- **Normal generation is near cost-free**: wllm trace-free completions run at 0.99× raw vLLM speed.
- **Token/logprob extraction is cheap**: adds ~12 ms (1.13×) for top_k=5 logprobs over 32 generated tokens. Postprocessing is ~1 ms.
- **Hidden-state replay is the main cost**: adds ~37-42 ms (1.41-1.45×) for middle-third layers due to the second vLLM encode pass on an isolated pooling runner.
- **Artifact format matters**: uncompressed NPZ serialization (~0.5 ms) is ~8× faster than compressed NPZ (~3.9 ms) but produces ~2× larger files. PT is comparably fast (~0.6 ms).
- **Online hidden states** avoid the second pass but require eager in-process generation, which changes the generation performance profile and produces numerically different activations.

Full benchmark data and methodology in `reports/wllm_vllm_stress_report.md` and `reports/wllm_online_vs_replay_report.md`.

## Private vLLM API Dependencies

wllm depends on several private (non-public) vLLM 0.10.2 APIs for hidden-state extraction. These are centralized in `src/runtime/vllm_compat.py` with version guards, expected shapes, and failure modes documented inline. The catalog includes:

| Private API | Used for |
|---|---|
| `LLM.apply_model(func)` | Installing scoped hooks on model workers |
| `LLM.encode(...)` | Replay hidden-state capture via pooling runner |
| `LLM.supported_tasks` | Runtime capability detection |
| Model executor internals | Hook installation and tensor capture |
| Pooling runner initialization | Isolated runner for replay extraction |
| Eager/in-process engine mode | Online hidden-state capture |

Each access point is version-guarded to vLLM 0.10.2. If the underlying private API changes in a future vLLM version, wllm raises a structured `UnsupportedExtractionError` (HTTP 501) rather than crashing with an opaque traceback. The full catalog with observed shapes and failure behavior is in the `vllm_compat.py` module docstring.

**Note for upgraders**: do not upgrade vLLM beyond 0.10.2 without auditing `vllm_compat.py`. The runtime will reject mismatched vLLM versions at import time with HTTP 503.

## Known Limitations

### Current limitations (0.1.0)

1. **No streaming.** `stream=true` is rejected on all endpoints. This is a design choice for the initial release; the response schema and error handling are simpler without streaming.
2. **Single GPU only.** `tensor_parallel_size=1` is required for hidden-state extraction. Multi-GPU tensor parallelism (sharded activations) is unsupported.
3. **Hidden states require ≤0.5 GPU memory.** The replay runner needs a second vLLM instance, consuming roughly double the model weights in VRAM. On a 24 GB GPU, this limits hidden-state replay to models under ~6B parameters.
4. **Online hidden states not numerically equivalent to replay.** Benchmarked relative L2 error up to 1.26 against replay-mode activations. These are different capture paths with different position semantics.
5. **`transformers` 5.12.1 compatibility.** The `all_special_tokens_extended` attribute removed in newer `transformers` causes errors when initializing the prewarmed pooling runner. Single-request hidden-state extraction is unaffected. Workaround: pin `transformers` to a vLLM 0.10.2-compatible version, or avoid `--prewarm-hidden-states`.
6. **Llama online hidden-state position mapping.** Online capture on Llama-3.2-3B fails because the captured tensor's sequence dimension does not cover the requested prompt positions. Replay capture works correctly for Llama.
7. **No attention weight extraction.** Fused attention backends (Flash Attention) do not expose weight matrices through the public vLLM path.
8. **No raw logits or exact entropy.** vLLM's public generation output exposes normalized logprobs, not the complete distribution.
9. **Limited vLLM option surface.** Many `vllm serve` options (quantization, LoRA, speculative decoding, multimodal, tool calling, structured outputs, etc.) are unsupported and rejected.
10. **Synchronous artifact writes.** `/v1/traces` writes JSON bundles and NPZ/PT artifacts synchronously on the request path. For large artifacts, this adds measurable latency.
11. **No authentication beyond static API key.** `--api-key` provides a single static bearer token. There is no OAuth, JWT, or multi-user access control.

### Not yet tested

- Models larger than 7B parameters.
- Batched requests (n > 1), which are explicitly rejected before extraction machinery is allocated.
- Non-Flash Attention backends.
- Windows or macOS platforms (developed and tested on Linux only).

## Compatibility with `vllm serve`

wllm is a drop-in replacement for **common** non-streaming vLLM OpenAI serving workflows, not a complete reimplementation. Supported options: `--host`, `--port`, `--dtype`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--tokenizer`, `--served-model-name`, `--api-key`, `--seed`, `--trust-remote-code`, `--local-files-only`, `--prewarm-hidden-states`, `--enable-online-hidden-states`.

Unsupported vLLM options (rejected with a clear error): `--quantization`, `--kv-cache-dtype`, `--device`, `--load-format`, `--pipeline-parallel-size`, `--speculative-model`, `--lora-modules`, `--chat-template`, `--enable-reasoning`, `--swap-space`, `--max-num-seqs`, `--max-num-batched-tokens`, `--download-dir`, `--enforce-eager`, `--disable-log-requests`, and many others. The full list is in the CLI help and in `tests/unit/test_cli_and_layout.py`.

## Error Envelope

All errors use an OpenAI-compatible envelope:

```json
{
  "error": {
    "message": "Human-readable description.",
    "type": "invalid_request_error",
    "param": null,
    "code": "schema_validation_failed",
    "details": {}
  }
}
```

Error types: `authentication_error` (401), `invalid_request_error` (422), `resource_limit_exceeded` (413), `unsupported_extraction` (501), `internal_server_error` (500), `runtime_unavailable` (503).

## Research Adapters

Research adapters consume generic trace objects. A `/v1/traces` response includes a `trace_manifest` that can be loaded:

```python
from artifacts import load_artifact
from research.token_baselines import TokenBaselineAdapter
from tracing.serialization import load_trace_bundle

trace = load_trace_bundle("./wllm-artifacts", trace_response["trace_manifest"])
result = TokenBaselineAdapter().run(trace)
```

Custom adapters depend on `TraceEnvelope` and artifact manifests, not server routes or paper-specific fields:

```python
from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class GeneratedTokenCount:
    name = "generated_token_count"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        start, end = trace.trace.spans["generated"]
        return ResearchResult(
            name=self.name,
            status="ok",
            values={"count": end - start},
        )
```

RAUQ, EigenScore, ActMap, and TokenBaseline adapters are under `src/research/`. All currently return `status="unsupported"` except TokenBaseline, which demonstrates the adapter protocol. Research adapters are never imported during normal serving; this isolation is verified by unit tests.

## Tests

**353 unit tests** (no GPU/vLLM required):

```bash
pytest tests/unit -q
```

**4 integration tests** (GPU + vLLM + local model required):

```bash
WLLM_TEST_MODEL=/path/to/local/model pytest tests/integration -m integration -v
```

The integration suite covers token/logprob extraction, replay hidden-state, online hidden-state, and repeated long-prompt hidden-state smoke tests. Reference configuration: vLLM 0.10.2, PyTorch 2.8.0+cu128, Qwen3-0.6B, `WLLM_TEST_MAX_MODEL_LEN=1024`, `WLLM_TEST_GPU_MEMORY_UTILIZATION=0.35`.

Multi-architecture validation results (Qwen3-0.6B, Llama-3.2-3B, Mistral-7B-Instruct-v0.3) are in `reports/wllm_multi_architecture_validation_report.md`. Benchmark results are in `reports/wllm_vllm_stress_report.md` and related report files.

## Capability Errors

The runtime never returns placeholder tensors or synthetic internals. Unsupported capabilities return HTTP 501 with an OpenAI-style error:

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

Error codes for common unsupported requests:
- `attention_weights_unavailable` — fused attention backends don't expose weights.
- `hidden_states_unavailable` — replay requirements not met (TP>1, GPU mem>0.5, no pooling runner).
- `online_hidden_states_disabled` — online capture requested without `--enable-online-hidden-states`.
- `exact_entropy_unavailable` — complete distribution not exposed by vLLM.
- `raw_logits_unavailable` — only normalized logprobs are available.
