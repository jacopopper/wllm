# w-llm

`w-llm` is a white-box inference server for uncertainty research. The serving
path is built around vLLM for high-throughput generation, with opt-in tracing
for token logprobs, entropy, selected hidden states, selected attention features,
and multi-sample uncertainty methods.

The normal OpenAI-compatible path stays trace-free by default:

```bash
wllm serve Qwen/Qwen3-0.6B --local-files-only
```

Endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/traces`
- `POST /v1/uncertainty`
- `GET /v1/trace-schema`

Tracing is enabled per request with a top-level `trace` object. OpenAI clients
can pass this through `extra_body={"trace": ...}`:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Give a one-line answer."}],
  "max_tokens": 32,
  "trace": {
    "hidden_states": {"layers": "middle", "positions": "last_generated"},
    "attentions": {"pattern": "previous_token", "layers": "middle_third"},
    "top_logprobs": 5
  }
}
```

Large hidden-state and ActMap payloads should be exported as artifacts rather
than returned inline. The JSON trace contract intentionally stores bounded,
selected slices.

## Development

The core schema and scorer modules are importable without GPU dependencies. For
the full server, install the project dependencies in an environment with vLLM:

```bash
python -m pip install -e ".[test,actmap]"
python -m pytest
```

