from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from schemas.artifacts import TraceBundleManifest
from schemas.traces import TraceEnvelope


class TraceLoadError(ValueError):
    """Raised when a persisted trace bundle cannot be trusted or decoded."""


def tensor_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    return {"name": name, "shape": list(array.shape), "dtype": str(array.dtype), "byte_size": int(array.nbytes)}


def load_trace_bundle(root: str | Path, manifest: TraceBundleManifest | Mapping[str, Any]) -> TraceEnvelope:
    bundle_manifest = TraceBundleManifest.model_validate(manifest)
    path = _trace_bundle_path(root, bundle_manifest.path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TraceLoadError(f"Trace bundle could not be read: {path}.") from exc
    if len(data) != bundle_manifest.byte_size:
        raise TraceLoadError(
            f"Trace bundle byte size mismatch: manifest={bundle_manifest.byte_size}, actual={len(data)}."
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != bundle_manifest.sha256:
        raise TraceLoadError("Trace bundle SHA-256 digest mismatch.")

    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceLoadError("Trace bundle is not valid UTF-8 JSON.") from exc
    try:
        trace = TraceEnvelope.model_validate(payload)
    except ValidationError as exc:
        raise TraceLoadError("Trace bundle payload does not match wllm.trace.v1.") from exc
    if trace.schema_version != bundle_manifest.schema_version:
        raise TraceLoadError(
            f"Trace schema version mismatch: "
            f"manifest={bundle_manifest.schema_version!r}, payload={trace.schema_version!r}."
        )
    if trace.id != bundle_manifest.trace_id:
        raise TraceLoadError(f"Trace ID mismatch: manifest={bundle_manifest.trace_id!r}, payload={trace.id!r}.")
    trace.trace_manifest = bundle_manifest
    return trace


def _trace_bundle_path(root: str | Path, relative_path: str) -> Path:
    if relative_path.startswith("/") or "\\" in relative_path:
        raise TraceLoadError("Trace bundle path must be a relative POSIX path.")
    root_path = Path(root).resolve()
    path = (root_path / relative_path).resolve()
    if root_path not in path.parents:
        raise TraceLoadError("Trace bundle path escapes the artifact root.")
    return path
