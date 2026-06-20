# wllm

OpenAI-compatible vLLM serving with opt-in white-box traces.

wllm runs normal chat and completion requests through vLLM, while adding a small
set of researcher-oriented endpoints for extracting token IDs, logprobs, hidden
states, and persisted tensor artifacts. Extraction is explicit: standard
generation requests do not allocate collectors, install hooks, or write
artifacts.

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
| Hidden states | conditional, single GPU |
| Attention weights | not supported |
| Tests | 361 unit tests plus GPU integration smoke tests |

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
| `--gpu-memory-utilization` | vLLM GPU memory utilization. Replay hidden states require `<= 0.5`. |
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

Replay capture is the default mode. It generates normally, then replays the final
token sequence through an isolated vLLM pooling runner with scoped hooks:

```bash
curl http://localhost:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Explain how transformer blocks process information.",
    "max_tokens": 32,
    "extract": {
      "hidden_states": [
        {
          "layers": "middle_third",
          "positions": "generated",
          "pool": "mean"
        }
      ]
    }
  }'
```

Replay requirements:

- `tensor_parallel_size=1`
- `gpu_memory_utilization <= 0.5`
- a vLLM model configuration that supports the pooling runner

Online capture is opt-in and captures from the active generation runner:

```bash
wllm serve Qwen/Qwen3-0.6B --enable-online-hidden-states
```

```json
{
  "extract": {
    "hidden_states": [
      {
        "layers": "middle",
        "positions": "prompt",
        "capture_mode": "online"
      }
    ]
  }
}
```

Online mode avoids the replay pass, but it starts the generation runner in eager
in-process mode with prefix caching disabled. It is not numerically
interchangeable with replay mode; choose the mode that matches your experiment.

### Selectors

| Selector | Accepted values |
|---|---|
| Layers | integer, integer list, negative indexes, `all`, `middle`, `middle_third` |
| Positions | integer, integer list, negative indexes, `prompt`, `generated`, `last`, `last_generated` |
| Attention key positions | position selector or `previous_token` |
| Hidden-state pooling | `null`, `mean`, `max`, `last` |

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

Takeaways:

- Normal generation stays near raw vLLM speed.
- Token and top-k logprob extraction are low overhead for small batch sizes.
- Replay hidden-state capture adds the cost of a second vLLM pass.
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

Research adapters live under `src/research/`. They consume `TraceEnvelope`
objects and artifact tensors; they do not add server routes or paper-specific
request fields.

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

Requests exceeding limits return HTTP 413 with an OpenAI-style error envelope.

## Error Handling

All API errors use an OpenAI-style envelope:

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

Common error classes:

| HTTP | Type | Examples |
|---:|---|---|
| 401 | `authentication_error` | Invalid or missing API key. |
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
--log-level
```

Not supported in `0.1.0`: streaming, quantization flags, LoRA, speculative
decoding, multimodal inputs, tool calling, structured outputs, reasoning parsers,
pipeline parallelism, custom chat templates, attention weight extraction, raw
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

Build package artifacts:

```bash
python -m build
```

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
- Attention weights are not exposed by the current vLLM path.
- Raw logits and exact entropy are not available from vLLM generation outputs.
- Artifact writes are synchronous on the request path.
- Authentication is limited to one static API key.
- Linux is the tested platform.

## License

MIT
