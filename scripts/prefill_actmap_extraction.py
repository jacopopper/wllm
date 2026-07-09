#!/usr/bin/env python3
"""
Example: Extracting prefill (prompt-only) activation trajectories / maps
using wllm's low-level hidden state capabilities.

Use case illustration: malicious / jailbreak prompt detection.
You process the prompt with max_tokens=0, capture hidden states
during prefill at desired layers and sites, then build custom
activation maps or features in your own research code.

wllm provides the raw capabilities:
- Request hidden_states with positions focused on the prompt.
- Use site="post_attn" / "post_mlp" / "block" for richer signals.
- Full per-token (pool=null) + artifacts for flexibility.
- Post-processing helpers (get_*, build_*) for maps without baking
  any specific method into the core.

This script does NOT implement any detection logic — that stays in
your research code.

Run against a running wllm server (with vllm extra):
  python scripts/prefill_actmap_extraction.py --prompt "..." --model ...
"""

import argparse
import os
from typing import Any

import httpx
import numpy as np

from research.features import get_prefill_activation_map, get_hidden_trajectories


def extract_prefill_map(
    base_url: str,
    prompt: str,
    model: str,
    layers: str = "middle_third",
    site: str = "block",
    max_inline: bool = False,
) -> np.ndarray | None:
    """Low-level extraction of prefill activation map."""
    client = httpx.Client(base_url=base_url, timeout=120.0)

    hidden_spec = {
        "layers": layers,
        "positions": "prompt",   # prefill only
        "pool": None,            # full per-token for maximum flexibility
        "site": site,
        "capture_mode": "replay",  # or "online" if enabled
    }

    artifacts = None
    if not max_inline:
        artifacts = {"format": "npz", "include": ["hidden_states"]}

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,   # pure prefill, no decoding
        "extract": {
            "hidden_states": [hidden_spec],
            "artifacts": artifacts,
        },
    }

    resp = client.post("/traces", json=body)  # or /extract for inline
    resp.raise_for_status()
    data = resp.json()

    artifact_root = "./wllm-artifacts"  # or whatever you configured
    artifact_dicts: dict[str, Any] = {}
    if "artifacts" in data:
        from artifacts import load_artifact
        for am in data["artifacts"]:
            artifact_dicts[am["path"]] = load_artifact(artifact_root, am)

    # Use the dedicated prefill helper (or get_hidden_trajectories + build yourself)
    actmap = get_prefill_activation_map(
        None,  # we will pass trace if needed; for simplicity use low-level load
        artifact_tensors=artifact_dicts,
        # The helper inside uses the trace, but for demo we show the idea
    )

    # More direct low-level path (recommended for full control):
    # Load the trace if using /traces
    if "trace_manifest" in data:
        from tracing.serialization import load_trace_bundle
        trace = load_trace_bundle(artifact_root, data["trace_manifest"])
        traj = get_hidden_trajectories(trace, artifact_tensors=artifact_dicts)
        prefill_traj = traj.get("prompt")
        if prefill_traj is not None:
            from research.features import build_activation_map
            actmap = build_activation_map(
                prefill_traj,
                num_layer_bins=32,
                num_dim_bins=128,
                # customize channel_specs for your detection features
            )

    return actmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("WLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layers", default="middle_third")
    parser.add_argument("--site", default="block", choices=["block", "post_attn", "post_mlp"])
    args = parser.parse_args()

    print("Extracting prefill activation map (low-level wllm capability)...")
    actmap = extract_prefill_map(
        args.base_url,
        args.prompt,
        args.model,
        layers=args.layers,
        site=args.site,
    )

    if actmap is not None:
        print("Got activation map with shape:", actmap.shape)
        print("You can now run your own malicious-prompt detector / probe on this tensor.")
        print("Example: feed to a small classifier, compute stats, etc.")
    else:
        print("No prefill map produced. Make sure the server supports hidden states and you requested artifacts.")


if __name__ == "__main__":
    main()
