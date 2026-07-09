#!/usr/bin/env python3
"""wllm dataset-building workflow example.

Demonstrates the full researcher workflow:
1. Read prompts from a JSONL file
2. Send each prompt to the wllm /v1/traces endpoint
3. Receive trace_manifest + artifact manifests
4. Load persisted trace bundles and artifact tensors
5. Run a research adapter on each trace

Usage:
    wllm serve Qwen/Qwen3-0.6B --local-files-only --port 8100 &
    python scripts/dataset_workflow.py --prompts prompts.jsonl --output results.jsonl

The script can also be used with an already-running server:
    WLLM_BASE_URL=http://localhost:8100/v1 python scripts/dataset_workflow.py --prompts data.jsonl

Requirements (install wllm with vllm extra):
    pip install -e '.[vllm,test]'

Input JSONL:
    {"id": "q1", "prompt": "Explain calibration briefly."}

Output JSONL fields:
    id, prompt, trace_id, token_count, generated_token_count, artifact_count,
    adapter_name, adapter_status, adapter_values, or error.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Client setup — uses httpx for full control over custom endpoints.
# ---------------------------------------------------------------------------


def _get_client(base_url: str | None = None):
    """Return an httpx client targeting the wllm server."""
    import httpx

    base_url = base_url or os.environ.get("WLLM_BASE_URL", "http://localhost:8100/v1")
    return httpx.Client(base_url=base_url, timeout=60.0)


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------


class PromptFileError(ValueError):
    """Raised when a prompt JSONL file cannot be used as a dataset input."""


def read_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Read prompts from a JSONL file.

    Each line must be a JSON object with at least a ``prompt`` key.
    Example line:
        {"prompt": "Explain the chain-of-thought reasoning process.", "id": "q1"}
    """
    prompt_path = Path(path)
    prompts: list[dict[str, Any]] = []
    try:
        fh = prompt_path.open(encoding="utf-8")
    except OSError as exc:
        raise PromptFileError(f"Could not read prompts file {prompt_path}: {exc}") from exc
    try:
        with fh:
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PromptFileError(
                        f"{prompt_path}:{line_num}: malformed JSONL entry: {exc.msg} at column {exc.colno}"
                    ) from exc
                if not isinstance(entry, dict):
                    raise PromptFileError(f"{prompt_path}:{line_num}: expected a JSON object")
                prompt = entry.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise PromptFileError(f"{prompt_path}:{line_num}: missing non-empty 'prompt' string")
                prompts.append(entry)
    except UnicodeDecodeError as exc:
        raise PromptFileError(f"{prompt_path}: prompt file is not valid UTF-8: {exc}") from exc
    return prompts


def extract_trace(
    client,
    prompt: str,
    model: str,
    max_tokens: int,
    include_logprobs: bool = True,
    include_hidden_states: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Send a single extraction request to /v1/traces.

    Returns the JSON response containing:
    - ``trace_manifest``: manifest for the persisted trace bundle JSON
    - ``artifacts``: list of artifact manifests (NPZ/PT files)
    """
    extract: dict[str, Any] = {"tokens": True}
    artifacts_include: list[str] = []

    if include_logprobs:
        extract["logprobs"] = {"top_k": top_k, "include_prompt": True}
        artifacts_include.append("logprobs")

    if include_hidden_states:
        # For ActMap-like activation maps: capture full per-token trajectories
        # for both prefill (prompt) and decoding (generated). Use pool=null for
        # per-position vectors; load via artifacts for large trajectories.
        extract["hidden_states"] = [
            {"layers": "middle_third", "positions": "prompt", "pool": None},      # prefill
            {"layers": "middle_third", "positions": "generated", "pool": None},  # decoding
        ]
        artifacts_include.append("hidden_states")

    if artifacts_include:
        extract["artifacts"] = {"format": "npz", "include": artifacts_include}

    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "extract": extract,
    }

    resp = client.post("/traces", json=body)
    resp.raise_for_status()
    return resp.json()


def error_message(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if code and message:
                return f"{code}: {message}"
            if message:
                return str(message)
    status_code = getattr(response, "status_code", None)
    return f"HTTP {status_code}: {exc}" if status_code is not None else str(exc)


def load_trace_and_artifacts(
    artifact_root: str,
    trace_response: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load a trace bundle and its associated artifacts from disk.

    Returns (TraceEnvelope, {artifact_path: tensor_dict})
    """
    from artifacts import load_artifact
    from tracing.serialization import load_trace_bundle

    # One-liner: load the trace bundle from its manifest
    trace = load_trace_bundle(artifact_root, trace_response["trace_manifest"])

    # One-liner: load each artifact from its manifest
    artifact_tensors: dict[str, Any] = {}
    for artifact_manifest in trace_response.get("artifacts", []):
        tensors = load_artifact(artifact_root, artifact_manifest)
        artifact_tensors[artifact_manifest["path"]] = tensors

    return trace, artifact_tensors


def run_adapter(trace, **options):
    """Run a research adapter on a loaded trace envelope.

    Replace with your own adapter (RAUQ, EigenScore, ActMap, or custom).
    The generic helpers in research.features make common UQ patterns easy
    (chosen_logprobs, last_token_hidden, etc.).
    """
    from research.token_baselines import TokenBaselineAdapter

    adapter = TokenBaselineAdapter()
    return adapter.run(trace, **options)


def build_result(
    prompt_entry: dict[str, Any],
    trace_response: dict[str, Any],
    trace,
    artifact_tensors: dict[str, Any],
    adapter_result,
) -> dict[str, Any]:
    """Assemble a structured result record."""
    token_ids = trace.trace.tokens.token_ids if trace.trace.tokens else []
    return {
        "id": prompt_entry.get("id"),
        "prompt": prompt_entry["prompt"][:120],
        "trace_id": trace.id,
        "token_count": len(token_ids),
        "generated_token_count": generated_token_count(trace),
        "artifact_count": len(artifact_tensors),
        "adapter_name": adapter_result.name,
        "adapter_status": adapter_result.status,
        "adapter_values": adapter_result.values,
    }


def generated_token_count(trace) -> int:
    generated_span = trace.trace.spans.get("generated", (0, 0))
    return max(0, generated_span[1] - generated_span[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid positive int value: {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive, got {parsed}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="wllm dataset-building workflow example")
    parser.add_argument("--prompts", required=True, help="Path to JSONL prompts file")
    parser.add_argument("--output", default="results.jsonl", help="Path to output JSONL results file")
    parser.add_argument("--model", default=os.environ.get("WLLM_MODEL", "Qwen/Qwen3-0.6B"), help="Model ID or path")
    parser.add_argument("--max-tokens", type=positive_int, default=64, help="Max tokens per generation")
    parser.add_argument("--artifact-dir", default="./wllm-artifacts", help="Directory for trace artifacts")
    parser.add_argument("--include-logprobs", dest="include_logprobs", action="store_true", help="Include logprob extraction")
    parser.add_argument("--no-logprobs", dest="include_logprobs", action="store_false", help="Extract token IDs only")
    parser.set_defaults(include_logprobs=True)
    parser.add_argument("--top-k", type=positive_int, default=5, help="Top-k logprobs to request when logprobs are enabled")
    parser.add_argument("--include-hidden-states", action="store_true", default=False, help="Include hidden-state extraction")
    args = parser.parse_args()

    # Step 1: Read prompts
    try:
        prompts = read_prompts(args.prompts)
    except PromptFileError as exc:
        print(f"Prompt file error: {exc}", file=sys.stderr)
        return 1
    if not prompts:
        print("No prompts loaded. Provide a JSONL file with at least one line containing a 'prompt' key.")
        return 1
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Step 2: Extract traces for each prompt
    client = _get_client()
    results: list[dict[str, Any]] = []

    try:
        for i, entry in enumerate(prompts):
            prompt_text = entry["prompt"]
            print(f"[{i + 1}/{len(prompts)}] {prompt_text[:60]}...", end=" ", flush=True)

            try:
                trace_response = extract_trace(
                    client,
                    prompt=prompt_text,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    include_logprobs=args.include_logprobs,
                    include_hidden_states=args.include_hidden_states,
                    top_k=args.top_k,
                )
            except Exception as exc:
                message = error_message(exc)
                print(f"FAILED: {message}")
                results.append({"id": entry.get("id"), "prompt": prompt_text[:120], "error": message})
                continue

            # Step 3: Load trace bundle and artifacts
            try:
                trace, artifact_tensors = load_trace_and_artifacts(args.artifact_dir, trace_response)
            except Exception as exc:
                message = error_message(exc)
                print(f"LOAD FAILED: {message}")
                results.append({"id": entry.get("id"), "prompt": prompt_text[:120], "error": f"load: {message}"})
                continue

            # Step 4: Run research adapter
            try:
                adapter_result = run_adapter(trace)
                result = build_result(entry, trace_response, trace, artifact_tensors, adapter_result)

                # Example of using new generic helpers for UQ-style analysis
                # (these work whether or not full logprobs/hidden were requested):
                try:
                    from research.features import chosen_logprobs, last_token_hidden
                    clp = chosen_logprobs(trace)
                    result["has_chosen_logprobs"] = bool(clp.get("generated"))
                    # last hidden useful for probing / spectral methods
                    # (loads from artifact if not inline)
                    _ = last_token_hidden(trace)  # best effort
                except Exception:
                    pass
            except Exception as exc:
                message = error_message(exc)
                print(f"ADAPTER FAILED: {message}")
                results.append(
                    {
                        "id": entry.get("id"),
                        "prompt": prompt_text[:120],
                        "trace_id": getattr(trace, "id", None),
                        "error": f"adapter: {message}",
                    }
                )
                continue

            print(f"-> {adapter_result.status} ({adapter_result.values})")
            results.append(result)
    finally:
        client.close()

    # Step 5: Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in results:
            json.dump(r, fh)
            fh.write("\n")
    print(f"Saved {len(results)} results to {output_path}")
    return 0


# ---------------------------------------------------------------------------
# Multi-sample helper (for EigenScore-style or Semantic Entropy workflows)
#
# Since extract endpoints enforce n=1, collect K traces for one prompt by
# calling multiple times (vary seed or temperature). The generic helpers
# above make it easy to stack last hiddens or chosen logprobs across samples.
#
# Example (not called automatically):
#   traces = []
#   arts_list = []
#   for s in range(k):
#       resp = extract_trace(client, prompt=..., model=..., max_tokens=..., seed=s)
#       tr, arts = load_trace_and_artifacts(artifact_dir, resp)
#       traces.append(tr)
#       arts_list.append(arts)
#   from research.features import stack_features_across_samples
#   hiddens = stack_features_across_samples(traces, feature="last_hidden", prefer_artifact_tensors_list=arts_list)
#   # hiddens.shape == (k, dim) ready for EigenScore etc.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(main())
