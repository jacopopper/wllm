from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from artifacts.npz import save_npz
from artifacts.torch import TorchArtifactUnavailableError, save_pt
from schemas.artifacts import ArtifactManifest, TraceBundleManifest
from server.errors import InvalidRequestError, UnsupportedExtractionError
from tracing.context import active_trace_id
from tracing.serialization import tensor_summary


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        *,
        trace_id: str,
        tensors: dict[str, Any],
        format: Literal["npz", "pt"] = "npz",
    ) -> ArtifactManifest:
        context_trace_id = active_trace_id.get()
        if context_trace_id is not None and context_trace_id != trace_id:
            raise InvalidRequestError(
                "Artifact trace_id does not match the active trace context.",
                code="trace_context_mismatch",
                param="trace_id",
                details={"active_trace_id": context_trace_id, "artifact_trace_id": trace_id},
            )
        artifact_id = f"art_{uuid.uuid4().hex}"
        self._ensure_root()
        suffix = ".npz" if format == "npz" else ".pt"
        path = self._artifact_path(f"{artifact_id}{suffix}")
        _, capture_dtypes = _tensor_manifest_metadata(tensors)
        if format == "npz":
            normalized = {name: _to_numpy_array(value) for name, value in tensors.items()}
            self._write_atomic(path, lambda temporary_path: save_npz(temporary_path, normalized))
            shapes, dtypes = _tensor_manifest_metadata(normalized)
        elif format == "pt":
            try:
                normalized = self._write_atomic(path, lambda temporary_path: save_pt(temporary_path, tensors))
            except TorchArtifactUnavailableError as exc:
                raise UnsupportedExtractionError(
                    "PT artifacts require torch to be installed.",
                    code="pt_artifacts_unavailable",
                    param="extract.artifacts.format",
                    details={"format": "pt"},
                ) from exc
            if normalized is None:
                normalized = tensors
            shapes, dtypes = _tensor_manifest_metadata(normalized)
        else:
            raise InvalidRequestError(f"Unsupported artifact format {format!r}.", param="extract.artifacts.format")
        data = path.read_bytes()
        return ArtifactManifest(
            artifact_id=artifact_id,
            format=format,
            path=path.relative_to(self.root).as_posix(),
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            included_tensor_names=list(tensors),
            tensor_shapes=shapes,
            tensor_dtypes=dtypes,
            tensor_capture_dtypes=capture_dtypes,
            tensor_storage_dtypes=dtypes,
            created=int(time.time()),
            trace_id=trace_id,
        )

    def put_trace_bundle(
        self,
        *,
        trace_id: str,
        schema_version: str,
        payload: dict[str, Any],
    ) -> TraceBundleManifest:
        context_trace_id = active_trace_id.get()
        if context_trace_id is not None and context_trace_id != trace_id:
            raise InvalidRequestError(
                "Trace bundle trace_id does not match the active trace context.",
                code="trace_context_mismatch",
                param="trace_id",
                details={"active_trace_id": context_trace_id, "bundle_trace_id": trace_id},
            )
        self._ensure_root()
        manifest_id = f"bundle_{uuid.uuid4().hex}"
        path = self._artifact_path(f"{manifest_id}.json")
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._write_atomic(path, lambda temporary_path: temporary_path.write_bytes(data))
        return TraceBundleManifest(
            manifest_id=manifest_id,
            schema_version=schema_version,
            path=path.relative_to(self.root).as_posix(),
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            created=int(time.time()),
            trace_id=trace_id,
        )

    def _artifact_path(self, name: str) -> Path:
        if "/" in name or "\\" in name or name.startswith("."):
            raise InvalidRequestError("Invalid artifact name.", code="invalid_artifact_path")
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise InvalidRequestError("Artifact path escapes the configured root.", code="invalid_artifact_path")
        return path

    def delete_manifest_path(self, relative_path: str) -> None:
        path = self._artifact_path(relative_path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _write_atomic(self, path: Path, writer: Callable[[Path], Any]) -> Any:
        temporary_path = self._artifact_path(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            result = writer(temporary_path)
            temporary_path.replace(path)
            return result
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise


def _tensor_manifest_metadata(tensors: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, str]]:
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    for name, value in tensors.items():
        shape, dtype = _shape_and_dtype(name, value)
        shapes[name] = shape
        dtypes[name] = dtype
    return shapes, dtypes


def _to_numpy_array(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        cpu_tensor = value.detach().cpu()
        try:
            return cpu_tensor.numpy()
        except TypeError:
            if getattr(cpu_tensor, "is_floating_point", lambda: False)():
                return cpu_tensor.float().numpy()
            raise
    return np.asarray(value)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "shape")


def _shape_and_dtype(name: str, value: Any) -> tuple[list[int], str]:
    if isinstance(value, np.ndarray):
        summary = tensor_summary(name, value)
        return summary["shape"], summary["dtype"]

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return [int(dim) for dim in shape], str(dtype)

    try:
        array = np.asarray(value)
    except Exception:
        return [], type(value).__name__
    summary = tensor_summary(name, array)
    return summary["shape"], summary["dtype"]
