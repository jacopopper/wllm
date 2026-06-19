from __future__ import annotations

from pathlib import Path
from typing import Any


class TorchArtifactUnavailableError(RuntimeError):
    pass


def save_pt(path: Path, tensors: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_pt_tensors(tensors)
    try:
        import torch
    except Exception as exc:
        raise TorchArtifactUnavailableError("torch is required for pt artifacts") from exc
    torch.save(payload, path)
    return payload


def normalize_pt_tensors(tensors: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        import numpy as np
    except Exception as exc:
        raise TorchArtifactUnavailableError("torch is required for pt artifacts") from exc
    return {name: _as_torch_tensor(value, torch, np) for name, value in tensors.items()}


def load_pt(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise TorchArtifactUnavailableError("torch is required for pt artifacts") from exc
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("pt artifact payload must be a dict of tensors")
    return data


def _as_torch_tensor(value: Any, torch: Any, np: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return torch.as_tensor(value)
