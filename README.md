# wllm

OpenAI-compatible vLLM serving with opt-in white-box traces.

wllm runs normal chat and completion requests through vLLM, while adding a small
set of researcher-oriented endpoints for extracting token IDs, logprobs, hidden
states (during both prefill and decoding, at richer internal sites), experimental
replay attention weights, raw logits, and persisted tensor artifacts.

Extraction is explicit: standard generation requests do not allocate collectors,
install hooks, run replay models, or write artifacts.

## Why wllm?

- Serve common OpenAI-compatible workloads with vLLM.
- Request model internals only when you need them.
- Build reproducible trace datasets with versioned schemas and artifact
  manifests.
- Keep unsupported internals explicit through structured capability errors
  instead of silent placeholders.

wllm is intended for AI safety and interpretability workflows where inference
speed matters, but token-level evidence and activation data also need to be
captured in a consistent format.

## Status

wllm is an early `0.1.0` release.

| Area | Status |
|---|---|
| Python | 3.10+ |
| vLLM | exactly `0.10.2` |
| OpenAI endpoints | `/v1/models`, `/v1/chat/completions`, `/v1/completions` |
| Extraction endpoints | `/v1/extract`, `/v1/traces`, `/v1/extraction-schema` |
| Streaming | not supported |
| Hidden states | conditional (prefill + decode supported); single GPU; richer capture sites (block, post-attn, post-mlp) |
| Attention weights | experimental, replay-only opt-in |
| Tests | unit tests plus GPU integration smoke tests |

The vLLM version is pinned intentionally. Hidden-state extraction relies on
private vLLM internals, all isolated in `src/runtime/vllm_compat.py`. Mismatched
vLLM versions are rejected at runtime with a structured error.

## Installation

For development and unit tests:

```bash
pip install -e '.[test]'
```

For vLLM-backed serving:

```bash
pip install -e '.[vllm]'
```

Optional PT artifact loading/saving uses PyTorch:

```bash
pip install -e '.[vllm,artifacts-pt]'
```

## Preflight

Before running long extraction jobs, check the environment explicitly:

1. Run the built-in preflight command:

   ```bash
   wllm doctor --model Qwen/Qwen3-0.6B --local-files-only
   ```

   `doctor` checks the Python environment, installed `wllm` metadata, exact
   vLLM version, optional PyTorch/Transformers packages, and local path-like
   model arguments. It does not initialize a model or download files.

2. Confirm the serving dependency set is installed:

   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

   The supported vLLM version is exactly `0.10.2`.

3. If the model is already cached or local, start with `--local-files-only` to
   avoid unexpected downloads.

4. For replay hidden states, use `--tensor-parallel-size 1` and configure
   `--gpu-memory-utilization 0.5` or lower so the isolated replay runner has
   room to initialize.

5. Query `/v1/extraction-schema` after startup and inspect `capabilities` and
   `limits` before sending large hidden-state or attention requests.

6. For attention replay, install the debug Transformers stack and start with
   `--enable-attention-weights`. Treat this path as experimental and prefer
   artifact-backed requests.

## Quickstart

Start a server:

```bash
wllm serve Qwen/Qwen3-0.6B
```

Use `--local-files-only` when the model is already cached and downloads should
be disabled.

Use the OpenAI Python client:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=16,
)

print(response.choices[0].message.content)
```

Or use curl:

```bash
curl http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Say hello:",
    "max_tokens": 16
  }'
```

## CLI

```bash
wllm serve MODEL [options]
```

Common options:

| Option | Description |
|---|---|
| `--host`, `--port` | Bind address. |
| `--dtype` | vLLM dtype, for example `auto`, `float16`, or `bfloat16`. |
| `--tensor-parallel-size` | Tensor parallel size. Hidden-state extraction currently requires `1`. |
| `--gpu-memory-utilization` | vLLM GPU memory utilization in `(0.0, 1.0]`. Replay hidden states require `<= 0.5`. |
| `--max-model-len` | Maximum model context length passed to vLLM. |
| `--tokenizer` | Optional tokenizer name or path. |
| `--served-model-name` | Model name returned by `/v1/models` and generation responses. |
| `--api-key` | Require `Authorization: Bearer ...` or a bare token. |
| `--seed` | Default seed for requests that omit a per-request seed. |
| `--trust-remote-code` | Forwarded to vLLM. |
| `--local-files-only` | Force Hugging Face offline mode before vLLM initialization. |
| `--artifact-dir` | Directory for trace bundles and tensor artifacts. |
| `--prewarm-hidden-states` | Initialize the replay hidden-state runner at startup. |
| `--enable-online-hidden-states` | Enable online hidden-state capture during generation. |
| `--enable-attention-weights` | Enable experimental replay-only attention extraction. |
| `--max-top-k` | Maximum `extract.logprobs.top_k`. |
| `--max-selected-layers` | Maximum selected layers per tensor request unless large extraction is enabled. |
| `--max-selected-heads` | Maximum selected attention heads per request unless large extraction is enabled. |
| `--max-selected-positions` | Maximum selected token positions per tensor request unless large extraction is enabled. |
| `--max-inline-tensor-bytes` | Maximum estimated tensor bytes returned inline in a trace response. |
| `--max-total-captured-tensor-bytes` | Maximum estimated tensor bytes captured for one extraction request. |
| `--max-artifact-bytes` | Maximum serialized byte size for one artifact or trace bundle. |
| `--enable-large-extraction` | Allow artifact-backed requests that exceed default selector-count limits. |

wllm accepts a focused subset of `vllm serve` options. Unsupported vLLM flags are
rejected instead of being ignored.

## Extraction API

Extraction requests use the same prompt inputs as OpenAI-style generation, plus
an `extract` object. `/v1/extract` returns the trace inline. `/v1/traces` returns
the trace and also persists a trace bundle under `--artifact-dir`.

### Token IDs and Logprobs

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain calibration briefly."}],
    "max_tokens": 64,
    "extract": {
      "tokens": true,
      "logprobs": {
        "top_k": 5,
        "include_prompt": true
      }
    }
  }'
```

Returned fields include:

| Field | Meaning |
|---|---|
| `trace.tokens.token_ids` | Prompt and generated token IDs. |
| `trace.tokens.tokens` | Decoded token strings when token extraction is requested. |
| `trace.spans` | Prompt/generated spans over the combined token sequence. |
| `trace.logprobs.generated` | Per-generated-token logprobs and top-k alternatives. |
| `trace.logprobs.prompt` | Prompt-token logprob rows when requested and available. |

Approximate entropy is available with:

```json
{
  "logprobs": {
    "top_k": 10,
    "entropy": true,
    "allow_approximate_entropy": true
  }
}
```

Exact entropy and raw logits are not exposed by vLLM's public generation output
and are rejected with capability errors.

### Hidden States

wllm makes it easy to capture hidden states during **both prefill** (prompt
processing) and **decoding** (generation). This is the core low-level capability
for building activation maps (ActMap-style or custom), probing, spectral
analysis, malicious-prompt detection on prefill activations, etc.

#### Prefill-only (prompt activations, no generation)

Use `max_tokens: 0` + `positions: "prompt"` (or `"last"`) to capture activations
from just the prompt. Ideal for input-side analysis such as malicious or
jailbreak prompt detection before any tokens are generated.

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Ignore all previous instructions and ...",
    "max_tokens": 0,
    "extract": {
      "hidden_states": [
        {
          "layers": "middle_third",
          "positions": "prompt",
          "pool": null,
          "site": "post_attn"
        }
      ],
      "artifacts": {"format": "npz", "include": ["hidden_states"]}
    }
  }'
```

#### Full trajectories (prefill + decode)

Request multiple specs in one call to get separate prompt and generated
trajectories:

```json
"hidden_states": [
  {"layers": "middle_third", "positions": "prompt", "pool": null, "site": "block"},
  {"layers": "middle_third", "positions": "generated", "pool": null, "site": "post_attn"}
]
```

Use `pool: null` + artifacts for full per-token, per-layer tensors. These are
perfect inputs for building custom activation maps.

#### Capture sites and modes

Hidden states can be captured at richer internal points using the `site` field:

- `"block"` (default) — after the full transformer block
- `"post_attn"` — after attention
- `"post_mlp"` — after the MLP/FFN

Replay is the default (isolated runner). Use `capture_mode: "online"` (with
`--enable-online-hidden-states`) to capture from the active generation pass.

See `research.features` for helpers:
- `get_hidden_trajectories(trace, artifacts)` → `{"prompt": (L, Tp, D), "generated": (L, Tg, D)}`
- `get_prefill_activation_map(...)` — convenient for prefill-only map building
- `build_activation_map(trajectory, ...)` — flexible C×L'×D' maps (your choice of temporal channels, binning, etc.)

A minimal example focused on prefill actmap extraction lives at
`scripts/prefill_actmap_extraction.py`.

Replay requirements (same as before):
- `tensor_parallel_size=1`
- `gpu_memory_utilization <= 0.5`
- Pooling runner supported by the model

Online mode runs the generation runner in eager mode with prefix caching
disabled. The two modes are not numerically interchangeable.

### Attention Weights

Attention extraction is experimental and disabled by default. When enabled, wllm
still generates normally with vLLM, then replays the final
`prompt_token_ids + generated_token_ids` through a separate Transformers/PyTorch
model with `output_attentions=True` and `use_cache=False`. The replay path
requires `torch` and `transformers`; missing dependencies are reported as
structured `attention_weights_unavailable` errors.

```bash
wllm serve Qwen/Qwen3-0.6B --enable-attention-weights
```

Artifact-backed attention traces are recommended because attention tensors scale
with sequence length squared:

```bash
curl http://localhost:8000/v1/traces \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Explain calibration briefly.",
    "max_tokens": 32,
    "extract": {
      "attentions": [
        {
          "layers": "middle",
          "heads": [0, 1],
          "query_positions": "generated",
          "key_positions": "previous_token"
        }
      ],
      "artifacts": {
        "format": "npz",
        "include": ["attentions"]
      }
    }
  }'
```

The replay path is memory-heavy and may differ from fused vLLM decode internals.
It is intended for bounded research traces, not low-latency serving. Online
attention capture from the active vLLM generation path is explicitly out of
scope.

### Selectors

| Selector | Accepted values |
|---|---|
| Layers | integer, integer list, negative indexes, `all`, `middle`, `middle_third` |
| Positions | integer, integer list, negative indexes, `prompt`, `generated`, `last`, `last_generated` |
| Hidden-state site | `"block"` (default), `"post_attn"`, `"post_mlp"` |
| Attention key positions | position selector or `previous_token` |
| Hidden-state pooling | `null`, `mean`, `max`, `last` |
| Hidden-state capture_mode | `"replay"` (default), `"online"` |

Use `positions: "prompt"` + `max_tokens: 0` for pure prefill activations. Use
multiple `hidden_states` objects (or combine with `"generated"`) to get
separate prefill and decoding trajectories for activation map construction.

## Artifacts

Use `/v1/traces` to persist trace bundles and tensor artifacts.

```bash
curl http://localhost:8000/v1/traces \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Summarize uncertainty:",
    "max_tokens": 32,
    "extract": {
      "tokens": true,
      "logprobs": {"top_k": 5},
      "hidden_states": [{"layers": "middle_third", "positions": "generated"}],
      "artifacts": {
        "format": "npz",
        "include": ["logprobs", "hidden_states"]
      }
    }
  }'
```

Artifacts are trace-scoped:

```text
wllm-artifacts/
└── traces/
    └── <trace_id>/
        ├── trace_bundle.json
        └── art_<id>.npz
```

Each manifest records path, byte size, SHA-256 digest, tensor names, shapes,
dtypes, capture dtypes, storage dtypes, and trace ID. Loaders verify digest and
path safety before returning tensors.

```python
from artifacts import load_artifact
from tracing.serialization import load_trace_bundle

trace = load_trace_bundle("./wllm-artifacts", response["trace_manifest"])
tensors = load_artifact("./wllm-artifacts", response["artifacts"][0])
```

Artifact formats:

| Format | Notes |
|---|---|
| `npz` | Default. Compressed unless `compression="uncompressed"` is requested. |
| `pt` | PyTorch format. Requires PyTorch and saves tensors loadable with modern `torch.load(..., weights_only=True)`. |

bf16 tensors are converted to float32 for NPZ storage with capture/storage dtype
metadata preserved in the manifest.

## Trace Schema

Trace responses use `wllm.trace.v1`.

| Section | Contents |
|---|---|
| `generation` | OpenAI-style generation summary with choices and usage. |
| `trace.tokens` | Token IDs and decoded tokens when requested. |
| `trace.spans` | Prompt/generated spans. |
| `trace.logprobs` | Selected logprobs and top-k alternatives. |
| `trace.hidden_states` | Tensor records or artifact references. |
| `trace.attentions` | Replay attention tensor records or artifact references. |
| `trace_manifest` | Persisted JSON bundle manifest for `/v1/traces`. |
| `artifacts` | Tensor artifact manifests. |
| `metadata` | Sampling params, capabilities, resolved selectors, and timings. |

## Capability Matrix

Validated on Linux with RTX 5090 24 GB, vLLM 0.10.2, PyTorch 2.8.0+cu128,
`max_model_len=1024`.

| Capability | Qwen3-0.6B | Llama-3.2-3B | Mistral-7B-Instruct-v0.3 |
|---|---:|---:|---:|
| Normal generation | yes | yes | yes |
| Chat completions | yes | base model has no template | yes |
| Text completions | yes | yes | yes |
| Token IDs | yes | yes | yes |
| Top-k logprobs | yes | yes | yes |
| Prompt logprobs | yes | yes | yes |
| Hidden-state replay | yes | yes | no, VRAM |
| Hidden-state online | yes | yes | no, VRAM |
| Prewarmed replay | yes | yes | no, VRAM |
| Attention replay | experimental | experimental | experimental, memory-heavy |
| NPZ artifacts | yes | yes | yes, token/logprob tested |
| PT artifacts | yes | yes | yes, token/logprob tested |

Notes:

- Llama-3.2-3B is a base model with no tokenizer chat template. Use
  `/v1/completions` with raw prompts or provide an instruct/chat model.
- Mistral-7B-Instruct-v0.3 is too large for hidden-state replay on a 24 GB GPU
  because replay requires a second model instance.

## Performance

Benchmark: Qwen3-0.6B, RTX 5090 24 GB, vLLM 0.10.2, `max_model_len=1024`,
`gpu_memory_utilization=0.35`, batch size 1, 163 prompt tokens plus 32 generated
tokens, warm runs.

| Mode | Median wall time | vs raw vLLM |
|---|---:|---:|
| Raw vLLM generate | 90.9 ms | 1.00x |
| wllm completion, trace-free | 90.0 ms | 0.99x |
| wllm extract tokens | 90.3 ms | 0.99x |
| wllm extract logprobs, top_k=5 | 102.3 ms | 1.13x |
| wllm trace + NPZ artifact | 103.8 ms | 1.14x |
| wllm hidden inline, 1 layer/1 position | 101.8 ms | 1.12x |
| wllm hidden NPZ, middle third | 132.2 ms | 1.45x |
| wllm hidden NPZ, uncompressed | 128.3 ms | 1.41x |
| wllm hidden PT | 128.2 ms | 1.41x |

Attention replay spot check: Qwen3-0.6B, RTX 5090 Laptop GPU, vLLM 0.10.2,
PyTorch 2.8.0+cu128, Transformers 5.12.1, `max_model_len=512`, bfloat16,
batch size 1, short prompt, 8 generated tokens, 3 measured runs.

| Mode | Median wall time | vs wllm completion | Generation ms | Replay capture ms | Serialization ms |
|---|---:|---:|---:|---:|---:|
| wllm completion, trace-free | 23.1 ms | 1.00x | - | - | - |
| Attention inline, 1 layer/2 heads/last query | 39.0 ms | 1.69x | 23.0 ms | 15.8 ms | 0.1 ms |
| Attention NPZ, 1 layer/2 heads/generated queries | 35.3 ms | 1.53x | 23.4 ms | 10.3 ms | 0.6 ms |

Takeaways:

- Normal generation stays near raw vLLM speed.
- Token and top-k logprob extraction are low overhead for small batch sizes.
- Replay hidden-state capture adds the cost of a second vLLM pass.
- Replay attention capture is no longer CPU-bound when CUDA is available, but it
  still adds a second Transformers forward pass and scales quadratically with
  sequence length.
- Uncompressed NPZ and PT reduce serialization latency at the cost of different
  storage tradeoffs.

## Dataset Workflow

`scripts/dataset_workflow.py` shows a complete prompt-to-trace workflow:

1. Read prompts from JSONL.
2. Call `/v1/traces`.
3. Load the persisted trace bundle.
4. Load tensor artifacts.
5. Run a research adapter.

```bash
wllm serve Qwen/Qwen3-0.6B --port 8100
python scripts/dataset_workflow.py \
  --prompts prompts.jsonl \
  --output results.jsonl \
  --model Qwen/Qwen3-0.6B
```

Each prompt file line must be a JSON object with a non-empty `prompt` string:

```json
{"id": "calibration-1", "prompt": "Explain calibration briefly."}
```

Malformed JSON, non-object rows, and missing or empty `prompt` fields stop the
workflow before any server requests are sent. Output rows include `id`,
`prompt`, `trace_id`, `token_count`, `generated_token_count`, `artifact_count`,
`adapter_name`, `adapter_status`, and `adapter_values`; failed rows include an
`error` field instead.

Research adapters live under `src/research/`. They consume `TraceEnvelope`
objects and artifact tensors; they do not add server routes or paper-specific
request fields.

Generic helpers for common patterns live in `research.features` (all operate on
`TraceEnvelope` + loaded artifacts):

- `get_hidden_trajectories(trace, artifacts)` → `{"prompt": (L, Tp, D), "generated": (L, Tg, D)}`
- `get_prefill_activation_map(...)` — convenient builder for prefill-only maps
- `build_activation_map(trajectory, num_layer_bins=..., num_dim_bins=..., channel_specs=...)` — fully flexible C×L'×D' maps
- `chosen_logprobs`, `last_token_hidden`, `entropy_from_raw_logits`, `stack_features_across_samples`, etc.

These are intentionally general building blocks. wllm does **not** implement
paper-specific methods (ActMap, EigenScore, RAUQ, etc.). Researchers bring their
own logic on top of the raw trajectories and maps.

Hidden-state extraction supports `site` ("block" | "post_attn" | "post_mlp")
for richer capture during **both prefill** (e.g. `positions="prompt"`, `max_tokens=0`)
**and decoding** (`positions="generated"`). Request multiple specs in one call
and use `pool: null` + artifacts to obtain full per-token, per-layer tensors.

This directly supports research such as:
- Activation-map (ActMap-style) construction from prefill activations
- Malicious/jailbreak prompt detection using only prompt-side hidden states
- Probing, spectral methods, and custom feature extraction on prefill + decode trajectories

See `scripts/prefill_actmap_extraction.py` for a focused, minimal example of
extracting prefill activations and building maps (your detection or analysis
logic stays completely outside wllm).

`research.actmap.ActMapAdapter` is provided as a small configurable example of
using the helpers above; it is deliberately general rather than a locked-in
implementation of any particular paper.

Raw logits and relaxed multi-sample (`n>1`) are also supported on extraction
endpoints.

A study of top white-box UQ/hallucination methods and the general capabilities
they benefit from is in `benchmarking/`.

## Resource Limits

Default server-side limits:

| Limit | Default |
|---|---:|
| `max_top_k` | 50 |
| `max_selected_layers` | 8 |
| `max_selected_heads` | 32 |
| `max_selected_positions` | 256 |
| `max_inline_tensor_bytes` | 1 MB |
| `max_total_captured_tensor_bytes` | 64 MB |
| `max_artifact_bytes` | 256 MB |
| `large_extraction_enabled` | false |

These defaults can be changed at server startup with the corresponding
`wllm serve` flags: `--max-top-k`, `--max-selected-layers`,
`--max-selected-heads`, `--max-selected-positions`,
`--max-inline-tensor-bytes`, `--max-total-captured-tensor-bytes`,
`--max-artifact-bytes`, and `--enable-large-extraction`. The active values are
reported by `/v1/extraction-schema` so client scripts can adapt before sending
large extraction requests.

Requests exceeding limits return HTTP 413 with an OpenAI-style error envelope.

The active vLLM generation runner is serialized inside one `VLLMRuntime`
instance. Concurrent HTTP requests are accepted by FastAPI, but calls that use
the shared generation runner are queued so vLLM scheduler state and online
hidden-state hooks cannot overlap. Replay hidden states, attention replay, and
artifact writes use their own scoped locks.

## Error Handling

All API errors use an OpenAI-style envelope:

```json
{
  "error": {
    "message": "Human-readable description.",
    "type": "invalid_request_error",
    "status": 422,
    "param": null,
    "code": "schema_validation_failed",
    "details": {}
  }
}
```

Common error classes:

| HTTP | Type | Examples |
|---:|---|---|
| 401 | `authentication_error` | Invalid or missing API key. |
| 404 | `invalid_request_error` | Unknown endpoint. |
| 405 | `invalid_request_error` | Wrong HTTP method for an endpoint. |
| 413 | `resource_limit_exceeded` | Extraction exceeds configured limits. |
| 422 | `invalid_request_error` | Schema errors, streaming requested, unsupported sampling field. |
| 501 | `unsupported_extraction` | Hidden states unavailable, attention weights unavailable, exact entropy unavailable. |
| 503 | `runtime_unavailable` | vLLM missing, unsupported vLLM version, model initialization failed. |

The runtime never returns fake tensors for unsupported internals.

## Compatibility with vLLM

wllm is a drop-in replacement for common non-streaming OpenAI-compatible
`vllm serve MODEL` workflows. It is not a complete vLLM server.

Supported CLI surface:

```text
doctor
serve
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
--prewarm-hidden-states
--enable-online-hidden-states
--enable-attention-weights
--max-top-k
--max-selected-layers
--max-selected-heads
--max-selected-positions
--max-inline-tensor-bytes
--max-total-captured-tensor-bytes
--max-artifact-bytes
--enable-large-extraction
--log-level
```

Not supported in `0.1.0`: streaming, quantization flags, LoRA, speculative
decoding, multimodal inputs, tool calling, structured outputs, reasoning parsers,
pipeline parallelism, custom chat templates, online attention capture, raw
logits, and exact entropy.

## Development

Run unit tests:

```bash
pytest tests/unit -q
```

Run integration tests with a local vLLM model:

```bash
WLLM_TEST_MODEL=/path/to/local/model \
WLLM_TEST_MAX_MODEL_LEN=1024 \
WLLM_TEST_GPU_MEMORY_UTILIZATION=0.35 \
pytest tests/integration -m integration -v
```

Run the smoke benchmark:

```bash
python scripts/benchmark_smoke.py /path/to/local/model --local-files-only
```

Plan or run the fuller latency suite:

```bash
python scripts/latency_suite.py --model /path/to/local/model --profile quick --dry-run
python scripts/latency_suite.py --model /path/to/local/model --profile quick --local-files-only
```

The latency suite compares raw vLLM generation, trace-free wllm completions,
token/logprob extraction, hidden-state capture, and experimental attention
replay cases. Generated latency and attention reports under `reports/` are local
benchmark outputs and are ignored by git unless intentionally curated.

Curated release evidence lives in:

- `reports/wllm_multi_architecture_validation_report.md`
- `reports/wllm_multi_architecture_validation_results.json`
- `reports/release-v0.1-readiness.md`

Build package artifacts:

```bash
python -m build
```

Release notes for the first usable version are in `CHANGELOG.md`.

## Project Layout

```text
src/
  cli.py                  # CLI entry point
  server/                 # FastAPI app, routes, error handling
  schemas/                # OpenAI, extraction, trace, artifact schemas
  runtime/                # vLLM runtime, orchestration, compatibility layer
  extractors/             # selector planning and extraction helpers
  artifacts/              # NPZ/PT persistence and safe loaders
  tracing/                # trace context and bundle loading
  research/               # trace-consuming research adapters
tests/
  unit/
  integration/
scripts/
```

## Limitations

- Streaming is rejected on all endpoints.
- Hidden-state extraction is single-GPU only.
- Replay hidden states require enough VRAM for a second vLLM runner.
- Online hidden states use a different execution path from replay and should not
  be treated as numerically equivalent.
- Attention weights are available only through opt-in Transformers replay, not
  from the current vLLM serving path.
- Raw logits and exact entropy are not available from vLLM generation outputs.
- Artifact writes are synchronous on the request path.
- Authentication is limited to one static API key.
- Linux is the tested platform.

## License

MIT
