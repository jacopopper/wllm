"""Generic, method-agnostic feature extraction helpers.

These operate on loaded TraceEnvelope objects (and optionally artifact tensors)
and return plain numpy arrays / lists. They are building blocks for researcher
code, not implementations of any specific paper method.

Intended usage:
    from research.features import chosen_logprobs

    logprobs = chosen_logprobs(trace)
    # logprobs["generated"] -> list[float] of chosen token logprobs
"""

from __future__ import annotations

from typing import Any

import numpy as np

from schemas.traces import TraceEnvelope


def chosen_logprobs(trace: TraceEnvelope, *, include_prompt: bool = False) -> dict[str, list[float]]:
    """Extract the sequence of chosen-token log probabilities from a trace.

    Prefers the first-class ``trace.trace.tokens.chosen_logprobs`` (populated
    for all extraction requests) and falls back to walking the detailed
    logprobs structure.

    Returns a dict with "generated" (always) and optionally "prompt".

    NaN is used for missing values.
    """
    result: dict[str, list[float]] = {}
    tokens = getattr(trace.trace, "tokens", None)
    if tokens is not None:
        clp = getattr(tokens, "chosen_logprobs", None) or (tokens.get("chosen_logprobs") if isinstance(tokens, dict) else None)
        if clp:
            prompt_len = 0
            spans = getattr(trace.trace, "spans", {}) or {}
            if isinstance(spans, dict):
                pspan = spans.get("prompt", (0, 0))
                prompt_len = max(0, pspan[1] - pspan[0]) if isinstance(pspan, (list, tuple)) else 0
            gen = [float(x) if x is not None else float("nan") for x in clp[prompt_len:]]
            result["generated"] = gen
            if include_prompt:
                result["prompt"] = [float(x) if x is not None else float("nan") for x in clp[:prompt_len]]
            return result

    # Fallback to detailed logprobs
    logprobs = getattr(trace.trace, "logprobs", {}) or {}
    for section in ("generated", "prompt"):
        if section == "prompt" and not include_prompt:
            continue
        rows = logprobs.get(section, []) or []
        seq: list[float] = []
        for row in rows:
            if isinstance(row, dict):
                lp = row.get("logprob")
            else:
                lp = getattr(row, "logprob", None)
            seq.append(float(lp) if lp is not None else float("nan"))
        if seq or section == "generated":
            result[section] = seq
    return result


def hidden_states_matrix(
    trace: TraceEnvelope,
    *,
    layer: int | None = None,
    prefer_artifact_tensors: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Best-effort extraction of hidden state matrices from the trace.

    Returns a dict like {"generated": (n_pos, dim) array, ...} for the requested
    (or first available) layer when possible.

    When tensors live only in artifacts, pass the loaded artifact dict(s) in
    `prefer_artifact_tensors` (mapping artifact path or id to the loaded dict).

    This helper is intentionally simple and generic; researchers doing serious
    work should select layers/positions explicitly at extraction time and load
    the corresponding artifact tensors.
    """
    out: dict[str, np.ndarray] = {}
    records = list(getattr(trace.trace, "hidden_states", []) or [])

    # Group by (layer, position_set) roughly; pick a representative layer if not specified
    candidates = []
    for rec in records:
        layers = getattr(rec, "layers", None) or (rec.get("layers") if isinstance(rec, dict) else None)
        if layers is None:
            continue
        if isinstance(layers, (list, tuple)):
            if layer is not None and layer not in layers:
                continue
            use_layer = layer if layer is not None else layers[0]
        else:
            use_layer = layers
            if layer is not None and use_layer != layer:
                continue
        positions = getattr(rec, "positions", None) or (rec.get("positions") if isinstance(rec, dict) else None)
        data = getattr(rec, "data", None)
        if data is None and prefer_artifact_tensors:
            # Try to resolve from artifact if referenced
            art_id = getattr(rec, "artifact_id", None) or (rec.get("artifact_id") if isinstance(rec, dict) else None)
            if art_id and art_id in prefer_artifact_tensors:
                # naive: look for any array whose name hints at hidden or the layer
                for _k, v in prefer_artifact_tensors[art_id].items():
                    arr = np.asarray(v)
                    if arr.ndim >= 2:
                        data = arr
                        break
        if data is not None:
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            candidates.append((use_layer, positions, arr))

    if not candidates:
        return out

    # Pick one (prefer the requested layer or the first)
    if layer is not None:
        for l, pos, arr in candidates:
            if l == layer:
                key = "generated" if (isinstance(pos, (list, tuple)) or pos == "generated") else "prompt"
                out[key] = arr
                break
    if not out:
        l, pos, arr = candidates[0]
        key = "generated" if (isinstance(pos, (list, tuple)) or str(pos) in ("generated", "last_generated")) else "prompt"
        out[key] = arr
    return out


def last_token_hidden(trace: TraceEnvelope, layer: int | str = "middle", *, prefer_artifact: dict[str, Any] | None = None) -> np.ndarray | None:
    """Convenience for the hidden vector of the last generated (or last prompt) token.

    Useful for SAPLMA-style probes and many UQ methods that use the final representation.

    Returns a 1D array or None.
    """
    mats = hidden_states_matrix(trace, layer=layer if isinstance(layer, int) else None, prefer_artifact_tensors=prefer_artifact)
    for key in ("generated", "prompt"):
        if key in mats:
            arr = np.asarray(mats[key])
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr[-1].astype(np.float32, copy=False)
            if arr.ndim == 1:
                return arr.astype(np.float32, copy=False)
    return None


def entropy_from_raw_logits(
    raw_logits: list[np.ndarray] | np.ndarray | dict[str, Any] | None,
    *,
    per_position: bool = False,
    temperature: float = 1.0,
) -> float | list[float]:
    """Compute exact (Shannon) entropy over the full vocabulary distribution
    from captured raw logits.

    This is the post-processing helper that becomes available once
    ``extract.logprobs.raw_logits=true`` (and artifacts are loaded if needed).

    Args:
        raw_logits: List of per-generated-step logits arrays (shape (vocab_size,)),
                    a single array, or an artifact dict containing "raw_logits".
        per_position: If True, return list of entropy per generated token.
                      Otherwise return the mean entropy across positions.
        temperature: Temperature to apply before softmax (default 1.0).

    Returns:
        Mean entropy (in nats) or list of per-position entropies.
        Uses NaN for missing/invalid positions.
    """
    if raw_logits is None:
        return float("nan") if not per_position else []

    if isinstance(raw_logits, dict):
        # support loading directly from artifact dict
        raw_logits = raw_logits.get("raw_logits") or raw_logits.get("logits") or raw_logits

    if isinstance(raw_logits, np.ndarray) and raw_logits.ndim == 1:
        raw_logits = [raw_logits]
    elif not isinstance(raw_logits, (list, tuple)):
        raw_logits = list(raw_logits) if raw_logits is not None else []

    entropies: list[float] = []
    for step_logits in raw_logits:
        if step_logits is None:
            entropies.append(float("nan"))
            continue
        arr = np.asarray(step_logits, dtype=np.float64)
        if arr.size == 0:
            entropies.append(float("nan"))
            continue
        # numerical stability
        arr = arr - np.max(arr)
        arr = arr / max(temperature, 1e-8)
        probs = np.exp(arr)
        probs = probs / (np.sum(probs) + 1e-12)
        # entropy in nats
        pos_ent = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(float(pos_ent))

    if per_position:
        return entropies
    valid = [e for e in entropies if not np.isnan(e)]
    return float(np.mean(valid)) if valid else float("nan")


def get_hidden_trajectories(
    trace: TraceEnvelope,
    artifact_tensors: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct per-phase hidden trajectories (L x T x D) from a trace.

    This is a key low-level capability for building custom activation maps
    (ActMap-style or other) over prefill and/or decoding phases.

    Prefill (prompt processing) and decoding activations are separated using
    the trace spans and the positions recorded in each TensorRecord.

    Returns:
        {
          "prompt": (L, num_prompt_tokens, D) or None,
          "generated": (L, num_generated_tokens, D) or None
        }

    Use `pool: null` in the extraction request for per-token vectors.
    Use artifacts for full trajectories (large data).

    This enables use cases like malicious prompt detection using only prefill
    activations (request hidden_states with positions="prompt", max_tokens=0).
    """
    if artifact_tensors is None:
        artifact_tensors = {}

    records = list(getattr(trace.trace, "hidden_states", []) or [])
    spans = getattr(trace.trace, "spans", {}) or {}
    p0, p1 = spans.get("prompt", (0, 0)) if isinstance(spans, dict) else (0, 0)
    g0, g1 = spans.get("generated", (0, 0)) if isinstance(spans, dict) else (0, 0)

    prompt_traj: dict[int, np.ndarray] = {}
    gen_traj: dict[int, np.ndarray] = {}

    for rec in records:
        layers = getattr(rec, "layers", None)
        if layers is None:
            layers = rec.get("layers") if isinstance(rec, dict) else None
        if layers is None:
            continue
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        layers = sorted(set(layers))

        pos = getattr(rec, "positions", None)
        if pos is None:
            pos = rec.get("positions") if isinstance(rec, dict) else None
        if pos is None:
            continue
        pos_list = list(pos) if isinstance(pos, (list, tuple, range)) else [pos]

        data = getattr(rec, "data", None)
        if data is None:
            # Look in artifacts. The name is hidden_states_{index} or similar
            for art_dict in artifact_tensors.values():
                if not isinstance(art_dict, dict):
                    continue
                for k, v in art_dict.items():
                    if "hidden" in k.lower():
                        data = v
                        break
                if data is not None:
                    break

        if data is None:
            continue

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # unlikely
        # Common case for pool=null: (num_selected_positions, D)
        # We assume the positions in this record correspond to the requested ones.

        is_prompt = any(p < g0 for p in pos_list) or (g0 == 0 and p1 > 0)  # rough
        is_gen = any(p >= g0 for p in pos_list)

        target = None
        if is_prompt and not is_gen:
            target = prompt_traj
            t_len = p1 - p0
        elif is_gen and not is_prompt:
            target = gen_traj
            t_len = g1 - g0
        else:
            # mixed or ambiguous; put in both or skip for simplicity
            continue

        # For multi-layer selection, the artifact tensor is often (L, T, D) or flattened.
        # Best effort reshape using known L and inferred T
        n_layers = len(layers)
        if arr.ndim == 2:
            # (T, D) or (L*T, D) etc. Try to reshape
            if arr.shape[0] == n_layers * (t_len or arr.shape[0] // n_layers):
                arr = arr.reshape(n_layers, -1, arr.shape[1])
            elif arr.shape[0] == t_len:
                arr = arr.reshape(1, t_len, -1)  # single layer case?
            # else leave as is and let user handle

        for li, layer in enumerate(layers):
            if arr.ndim == 3:
                layer_arr = arr[li]
            else:
                layer_arr = arr  # fallback
            target[layer] = layer_arr

    result = {}
    for phase, traj_dict in [("prompt", prompt_traj), ("generated", gen_traj)]:
        if traj_dict:
            sorted_layers = sorted(traj_dict.keys())
            stacked = np.stack([traj_dict[l] for l in sorted_layers], axis=0)
            result[phase] = stacked
        else:
            result[phase] = None
    return result


def build_activation_map(
    trajectory: np.ndarray,
    *,
    num_layer_bins: int = 32,
    num_dim_bins: int = 128,
    channel_specs: list[str] | None = None,
) -> np.ndarray:
    """Build a fixed-size activation map from a (L, T, D) or (L, D) trajectory.

    This provides a flexible, ActMap-like post-processing over generic hidden
    state artifacts without hard-coding any specific paper method.

    Default channels (if channel_specs=None) implement common temporal summaries
    inspired by activation-trajectory methods (segment means, final state,
    dynamics, etc.).

    Returns shape (C, L', D') after adaptive pooling on layer and dim axes.
    """
    arr = np.asarray(trajectory, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]  # (L, 1, D)
    if arr.ndim != 3:
        raise ValueError("trajectory must be (L, T, D) or (L, D)")

    L, T, D = arr.shape

    if channel_specs is None:
        channel_specs = [
            "first_quarter_mean",
            "second_quarter_mean",
            "third_quarter_mean",
            "fourth_quarter_mean",
            "final_token",
            "final_window_mean",
            "token_std",
            "token_max",
            "last_minus_first",
            "temporal_slope",
            "layer_rms",
            "abs_step_diff",
        ]

    channels = []
    for spec in channel_specs:
        if spec == "first_quarter_mean":
            ch = np.mean(arr[:, : max(1, T//4), :], axis=1)
        elif spec == "second_quarter_mean":
            s, e = T//4, 2*T//4
            ch = np.mean(arr[:, s:e, :], axis=1)
        elif spec == "third_quarter_mean":
            s, e = 2*T//4, 3*T//4
            ch = np.mean(arr[:, s:e, :], axis=1)
        elif spec == "fourth_quarter_mean":
            ch = np.mean(arr[:, 3*T//4:, :], axis=1)
        elif spec == "final_token":
            ch = arr[:, -1, :]
        elif spec == "final_window_mean":
            w = min(8, T)
            ch = np.mean(arr[:, -w:, :], axis=1)
        elif spec == "token_std":
            ch = np.std(arr, axis=1)
        elif spec == "token_max":
            ch = np.max(arr, axis=1)
        elif spec == "last_minus_first":
            ch = arr[:, -1, :] - arr[:, 0, :]
        elif spec == "temporal_slope":
            # simple linear fit slope per (layer, dim)
            ts = np.arange(T, dtype=np.float32)
            # (L, D, T) @ ts -> slope approx
            ch = np.tensordot(arr, ts, axes=([1], [0])) / (T * np.sum(ts**2) - np.sum(ts)**2 + 1e-8)  # rough
            # simpler: use np.polyfit per, but vectorized approx with cov
            mean_t = np.mean(ts)
            cov = np.mean((ts - mean_t) * (arr - np.mean(arr, axis=1, keepdims=True)), axis=1)
            var_t = np.var(ts)
            ch = cov / (var_t + 1e-8)
        elif spec == "layer_rms":
            ch = np.sqrt(np.mean(arr**2, axis=1))
        elif spec == "abs_step_diff":
            if T > 1:
                diffs = np.abs(arr[:, 1:, :] - arr[:, :-1, :])
                ch = np.mean(diffs, axis=1)
            else:
                ch = np.zeros((L, D), dtype=np.float32)
        else:
            raise ValueError(f"Unknown channel spec: {spec}")
        channels.append(ch)

    map_tensor = np.stack(channels, axis=0)  # (C, L, D)

    # Adaptive pool layer axis to num_layer_bins, dim to num_dim_bins
    # Use simple binning mean (adaptive average pool)
    def adaptive_pool(x: np.ndarray, out_bins: int, axis: int) -> np.ndarray:
        n = x.shape[axis]
        if n == out_bins:
            return x
        edges = np.linspace(0, n, out_bins + 1, dtype=int)
        pooled = []
        for i in range(out_bins):
            sl = slice(edges[i], edges[i+1])
            if axis == 0:
                pooled.append(np.mean(x[sl], axis=0))
            elif axis == 1:
                pooled.append(np.mean(x[:, sl], axis=1))
            else:
                pooled.append(np.mean(x[:, :, sl], axis=2))
        return np.stack(pooled, axis=axis)

    # pool layer (axis 1), then dim (axis 2)
    map_tensor = adaptive_pool(map_tensor, num_layer_bins, axis=1)
    map_tensor = adaptive_pool(map_tensor, num_dim_bins, axis=2)

    # standardize per channel/layer/dim ? optional, user can do
    return map_tensor.astype(np.float32)


def get_prefill_activation_map(
    trace: TraceEnvelope,
    artifact_tensors: dict[str, Any] | None = None,
    *,
    layers: list[int] | str | None = "middle_third",
    num_layer_bins: int = 32,
    num_dim_bins: int = 128,
    channel_specs: list[str] | None = None,
    site: str | None = None,  # informational
) -> np.ndarray | None:
    """Convenience for extracting a pooled activation map from *prefill only*
    hidden states.

    This is particularly useful for input-side analysis such as malicious
    prompt detection, where you want signals from processing the prompt
    before any tokens are generated.

    Request example (low-level capability):
        extract={
            "hidden_states": [{
                "layers": layers,
                "positions": "prompt",   # or "last" for the final prompt token
                "pool": null,
                "site": site or "block"
            }],
            "artifacts": {"include": ["hidden_states"]}
        },
        max_tokens=0

    Then pass the loaded artifacts here.
    """
    traj = get_hidden_trajectories(trace, artifact_tensors)
    prefill = traj.get("prompt")
    if prefill is None or prefill.size == 0:
        return None
    return build_activation_map(
        prefill,
        num_layer_bins=num_layer_bins,
        num_dim_bins=num_dim_bins,
        channel_specs=channel_specs,
    )


def stack_features_across_samples(
    traces: list[TraceEnvelope],
    *,
    feature: str = "last_hidden",
    layer: int | str = "middle",
    prefer_artifact_tensors_list: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Small helper to stack features across multiple samples (for EigenScore,
    consistency UQ, etc.).

    Currently supports "last_hidden" (returns K x D array).
    """
    if prefer_artifact_tensors_list is None:
        prefer_artifact_tensors_list = [None] * len(traces)

    stacked = []
    for trace, arts in zip(traces, prefer_artifact_tensors_list or [None] * len(traces)):
        if feature == "last_hidden":
            vec = last_token_hidden(trace, layer=layer, prefer_artifact=arts)
            if vec is not None:
                stacked.append(vec)
        # extend with other features as needed
    if not stacked:
        return np.array([])
    return np.stack(stacked)
