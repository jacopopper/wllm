from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactTensorInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str


class ArtifactManifest(BaseModel):
    artifact_id: str
    format: Literal["npz", "pt"]
    compression: Literal["compressed", "uncompressed"] | None = None
    path: str
    byte_size: int = Field(ge=0)
    sha256: str
    included_tensor_names: list[str]
    tensor_shapes: dict[str, list[int]]
    tensor_dtypes: dict[str, str]
    tensor_capture_dtypes: dict[str, str] = Field(default_factory=dict)
    tensor_storage_dtypes: dict[str, str] = Field(default_factory=dict)
    created: int
    trace_id: str


class TraceBundleManifest(BaseModel):
    manifest_id: str
    object: Literal["wllm.trace_manifest"] = "wllm.trace_manifest"
    schema_version: str
    path: str
    byte_size: int = Field(ge=0)
    sha256: str
    created: int
    trace_id: str
