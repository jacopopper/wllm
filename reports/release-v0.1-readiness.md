# wllm 0.1.0 Release Readiness Note

Created: 2026-06-27

This note classifies the current release worktree and records the non-destructive
validation gates for the first safety-researcher-usable release.

## Worktree Classification

Keep as release source:

- `CHANGELOG.md`
- `first-version-safety-researcher-library-plan.md`
- `scripts/latency_suite.py`
- CLI, runtime, schema, server, artifact, research, and unit-test updates

Keep as curated release evidence:

- `reports/wllm_multi_architecture_validation_report.md`
- `reports/wllm_multi_architecture_validation_results.json`

Ignore as generated local benchmark output:

- `reports/wllm_latency_*.json`
- `reports/wllm_latency_*.md`
- `reports/wllm_attention_*.json`
- `reports/wllm_attention_*.md`
- `reports/*stress*.json`
- `reports/*stress*.md`
- `reports/wllm_online_vs_replay_*`
- `wllm-artifacts/`
- `wllm-stress-artifacts/`
- `wllm-latency-artifacts/`

Leave untouched unless explicitly curated later:

- Local cache directories such as `.venv/`, `.pytest_cache/`, `.mypy_cache/`,
  and `__pycache__/`
- `uv.lock`, which is currently ignored by project policy
- Local stress harnesses ignored in `.gitignore`

## Validation Completed

- `uv run python -m pytest tests/unit -q`
  - Result: 411 passed, 1 Starlette/httpx deprecation warning.
- `uv run python -m pytest tests/integration -q`
  - Result: 4 skipped because `WLLM_TEST_MODEL` was not set for this run.
- `WLLM_TEST_MODEL=/home/jacopodardini/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca WLLM_TEST_MAX_MODEL_LEN=1024 WLLM_TEST_GPU_MEMORY_UTILIZATION=0.35 uv run python -m pytest tests/integration -m integration -v`
  - Result: 4 passed against local Qwen3-0.6B, covering token/logprob
    extraction, replay hidden states, online hidden states, and repeated
    full-sequence replay. Warnings were limited to upstream SWIG/fork/NCCL
    deprecation/shutdown warnings.
- `.venv/bin/python -m build --sdist --wheel --outdir /tmp/wllm-build-check-review-fixes`
  - Result: built `wllm-0.1.0.tar.gz` and `wllm-0.1.0-py3-none-any.whl`.
- `git diff --check`
  - Result: clean.
- `.venv/bin/python -m zipfile -l /tmp/wllm-build-check-review-fixes/wllm-0.1.0-py3-none-any.whl`
  - Result: wheel contains the flat-layout modules and `entry_points.txt`.
- `tar -tf /tmp/wllm-build-check-review-fixes/wllm-0.1.0.tar.gz`
  - Result: sdist includes `CHANGELOG.md`, README, scripts, source, tests, and
    exactly the curated reports listed above.
- `uv run python -m pytest tests/unit/test_api_compliance.py tests/unit/test_errors.py tests/unit/test_app.py -q`
  - Result: 58 passed, including 404/405 OpenAI-style error envelopes and
    constant-time API key comparison coverage.
- Post-review fixes from the v0.1 readiness pass:
  - Dataset workflow adapter/result-building failures are captured as per-prompt
    error rows so prior batch results are still saved.
  - Redundant hidden-state preflight state and unused hidden tensor pool storage
    were removed.
- `uv run python scripts/dataset_workflow.py --help`
  - Result: help renders successfully.
- `uv run python scripts/latency_suite.py --model fake --dry-run --profile quick`
  - Result: dry-run lists the expected benchmark jobs without loading models.
- Manual HTTP quickstart smoke through `wllm serve` on local Qwen3-0.6B:
  - `GET /v1/models`: 200.
  - `POST /v1/completions`: 200 and generated text.
  - `POST /v1/chat/completions`: 200 and OpenAI-style chat response.
  - Concurrent `POST /v1/extract` and `POST /v1/traces`: both 200 after
    serializing access to the shared vLLM generation runner.
  - Public loaders `load_trace_bundle(...)` and `load_artifact(...)` loaded the
    persisted trace/artifact.
  - `TokenBaselineAdapter().run(trace)`: `ok`.
- README JSON validation
  - Result: 4 fenced JSON examples and 5 curl request bodies parse as JSON.
- `wllm doctor --json` against the cached Qwen3-0.6B snapshot
  - Result: Python, `wllm`, vLLM 0.10.2, torch, transformers, and model path all
    report `ok`.

## Pending Validation

No required release gate is pending in this note.

Re-run the live local-model smoke with:

```bash
WLLM_TEST_MODEL=/home/jacopodardini/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
WLLM_TEST_MAX_MODEL_LEN=1024 \
WLLM_TEST_GPU_MEMORY_UTILIZATION=0.35 \
uv run python -m pytest tests/integration -m integration -v
```
