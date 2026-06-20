from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from artifacts.npz import load_npz
from artifacts.torch import TorchArtifactUnavailableError, load_pt
from schemas.artifacts import ArtifactManifest


class ArtifactLoadError(ValueError):
    """Raised when an artifact cannot be trusted or decoded."""


def load_artifact(root: str | Path, manifest: ArtifactManifest | Mapping[str, Any]) -> dict[str, Any]:
    artifact_manifest = ArtifactManifest.model_validate(manifest)
    path = _artifact_path(root, artifact_manifest.path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactLoadError(f"Artifact could not be read: {path}.") from exc
    if len(data) != artifact_manifest.byte_size:
        raise ArtifactLoadError(
            f"Artifact byte size mismatch: manifest={artifact_manifest.byte_size}, actual={len(data)}."
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != artifact_manifest.sha256:
        raise ArtifactLoadError("Artifact SHA-256 digest mismatch.")

    if artifact_manifest.format == "npz":
        tensors = _load_npz_artifact(path)
    elif artifact_manifest.format == "pt":
        tensors = _load_pt_artifact(path)
    else:
        raise ArtifactLoadError(f"Unsupported artifact format {artifact_manifest.format!r}.")
    _validate_tensor_manifest(tensors, artifact_manifest)
    return tensors


def _artifact_path(root: str | Path, relative_path: str) -> Path:
    if relative_path.startswith("/") or "\\" in relative_path:
        raise ArtifactLoadError("Artifact path must be a relative POSIX path.")
    root_path = Path(root).resolve()
    path = (root_path / relative_path).resolve()
    if root_path not in path.parents:
        raise ArtifactLoadError("Artifact path escapes the artifact root.")
    return path


def _load_npz_artifact(path: Path) -> dict[str, Any]:
    try:
        return load_npz(path)
    except Exception as exc:
        raise ArtifactLoadError("NPZ artifact could not be decoded.") from exc


def _load_pt_artifact(path: Path) -> dict[str, Any]:
    try:
        return load_pt(path)
    except TorchArtifactUnavailableError as exc:
        raise ArtifactLoadError("PT artifacts require torch to be installed.") from exc
    except Exception as exc:
        raise ArtifactLoadError("PT artifact could not be decoded.") from exc


def _validate_tensor_manifest(tensors: dict[str, Any], manifest: ArtifactManifest) -> None:
    expected_names = set(manifest.included_tensor_names)
    actual_names = set(tensors)
    if actual_names != expected_names:
        raise ArtifactLoadError(
            f"Artifact tensor names mismatch: manifest={sorted(expected_names)!r}, actual={sorted(actual_names)!r}."
        )
    for name, tensor in tensors.items():
        shape, dtype = _shape_and_dtype(tensor)
        if shape != manifest.tensor_shapes.get(name):
            raise ArtifactLoadError(
                f"Artifact tensor {name!r} shape mismatch: "
                f"manifest={manifest.tensor_shapes.get(name)!r}, actual={shape!r}."
            )
        expected_dtype = manifest.tensor_storage_dtypes.get(name) or manifest.tensor_dtypes.get(name)
        if dtype != expected_dtype:
            raise ArtifactLoadError(
                f"Artifact tensor {name!r} dtype mismatch: manifest={expected_dtype!r}, actual={dtype!r}."
            )


def _shape_and_dtype(value: Any) -> tuple[list[int], str]:
    if isinstance(value, np.ndarray):
        return list(value.shape), str(value.dtype)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return [int(dim) for dim in shape], str(dtype)
    array = np.asarray(value)
    return list(array.shape), str(array.dtype)
