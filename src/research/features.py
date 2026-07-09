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
