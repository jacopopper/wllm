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
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Client setup — uses httpx for full control over custom endpoints.
# ---------------------------------------------------------------------------


def _get_client():
    """Return an httpx client targeting the wllm server."""
    import httpx

    base_url = os.environ.get("WLLM_BASE_URL", "http://localhost:8100/v1")
    return httpx.Client(base_url=base_url, timeout=60.0)


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------


def read_prompts(path: str) -> list[dict[str, Any]]:
    """Read prompts from a JSONL file.

    Each line must be a JSON object with at least a ``prompt`` key.
    Example line:
        {"prompt": "Explain the chain-of-thought reasoning process.", "id": "q1"}
    """
    prompts: list[dict[str, Any]] = []
    with open(path) as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {line_num}: {exc}")
                continue
            if "prompt" not in entry:
                print(f"Skipping line {line_num}: missing 'prompt' key")
                continue
            prompts.append(entry)
    return prompts


def extract_trace(
    client,
    prompt: str,
    model: str,
    max_tokens: int,
    artifact_dir: str,
    include_logprobs: bool = True,
    include_hidden_states: bool = False,
) -> dict[str, Any]:
    """Send a single extraction request to /v1/traces.

    Returns the JSON response containing:
    - ``trace_manifest``: manifest for the persisted trace bundle JSON
    - ``artifacts``: list of artifact manifests (NPZ/PT files)
    """
    extract: dict[str, Any] = {"tokens": True}
    artifacts_include: list[str] = []

    if include_logprobs:
        extract["logprobs"] = {"top_k": 5, "include_prompt": True}
        artifacts_include.append("logprobs")

    if include_hidden_states:
        extract["hidden_states"] = [{"layers": "middle", "positions": "last_generated"}]
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
        "generated_token_count": max(0, trace.trace.spans.get("generated", (0, 0))[1] - trace.trace.spans.get("generated", (0, 0))[0]),
        "artifact_count": len(artifact_tensors),
        "adapter_name": adapter_result.name,
        "adapter_status": adapter_result.status,
        "adapter_values": adapter_result.values,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="wllm dataset-building workflow example")
    parser.add_argument("--prompts", required=True, help="Path to JSONL prompts file")
    parser.add_argument("--output", default="results.jsonl", help="Path to output JSONL results file")
    parser.add_argument("--model", default=os.environ.get("WLLM_MODEL", "Qwen/Qwen3-0.6B"), help="Model ID or path")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per generation")
    parser.add_argument("--artifact-dir", default="./wllm-artifacts", help="Directory for trace artifacts")
    parser.add_argument("--include-logprobs", action="store_true", default=True, help="Include logprob extraction")
    parser.add_argument("--include-hidden-states", action="store_true", default=False, help="Include hidden-state extraction")
    args = parser.parse_args()

    # Step 1: Read prompts
    prompts = read_prompts(args.prompts)
    if not prompts:
        print("No prompts loaded. Provide a JSONL file with at least one line containing a 'prompt' key.")
        return
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Step 2: Extract traces for each prompt
    client = _get_client()
    results: list[dict[str, Any]] = []

    for i, entry in enumerate(prompts):
        prompt_text = entry["prompt"]
        print(f"[{i + 1}/{len(prompts)}] {prompt_text[:60]}...", end=" ", flush=True)

        try:
            trace_response = extract_trace(
                client,
                prompt=prompt_text,
                model=args.model,
                max_tokens=args.max_tokens,
                artifact_dir=args.artifact_dir,
                include_logprobs=args.include_logprobs,
                include_hidden_states=args.include_hidden_states,
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"id": entry.get("id"), "prompt": prompt_text[:120], "error": str(exc)})
            continue

        # Step 3: Load trace bundle and artifacts
        try:
            trace, artifact_tensors = load_trace_and_artifacts(args.artifact_dir, trace_response)
        except Exception as exc:
            print(f"LOAD FAILED: {exc}")
            results.append({"id": entry.get("id"), "prompt": prompt_text[:120], "error": f"load: {exc}"})
            continue

        # Step 4: Run research adapter
        adapter_result = run_adapter(trace)
        print(f"→ {adapter_result.status} ({adapter_result.values})")

        result = build_result(entry, trace_response, trace, artifact_tensors, adapter_result)
        results.append(result)

    client.close()

    # Step 5: Save results
    output_path = Path(args.output)
    with output_path.open("w") as fh:
        for r in results:
            json.dump(r, fh)
            fh.write("\n")
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
