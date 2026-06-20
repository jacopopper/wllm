from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.artifacts import ArtifactManifest, TraceBundleManifest

TRACE_SCHEMA_VERSION: Literal["wllm.trace.v1"] = "wllm.trace.v1"


class TensorRecord(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    capture_dtype: str | None = None
    storage_dtype: str | None = None
    device: str
    layers: list[int] | None = None
    positions: list[int] | None = None
    heads: list[int] | None = None
    capture_site: str
    capture_mode: Literal["replay", "online"] | None = None
    capture_phase: str | None = None
    position_semantics: dict[str, Any] = Field(default_factory=dict)
    capture_metadata: dict[str, Any] = Field(default_factory=dict)
    data: Any | None = None
    artifact_id: str | None = None
    byte_size: int | None = None


class TokenTrace(BaseModel):
    token_ids: list[int] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)


class TraceData(BaseModel):
    tokens: TokenTrace | dict[str, Any] = Field(default_factory=dict)
    spans: dict[str, tuple[int, int]] = Field(default_factory=dict)
    logprobs: dict[str, Any] = Field(default_factory=dict)
    hidden_states: list[TensorRecord] = Field(default_factory=list)
    attentions: list[TensorRecord] = Field(default_factory=list)


class TimingMetadata(BaseModel):
    queue: float = 0.0
    generation: float = 0.0
    capture: float = 0.0
    postprocess: float = 0.0
    serialization: float = 0.0
    extraction_overhead: float = 0.0
    total: float = 0.0


class TraceMetadata(BaseModel):
    sampling: dict[str, Any] = Field(default_factory=dict)
    resolved_selectors: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    capture: dict[str, Any] = Field(default_factory=dict)
    timing_ms: TimingMetadata = Field(default_factory=TimingMetadata)


class TraceEnvelope(BaseModel):
    schema_version: Literal["wllm.trace.v1"] = TRACE_SCHEMA_VERSION
    id: str
    object: Literal["wllm.trace"] = "wllm.trace"
    created: int
    model: str
    generation: dict[str, Any]
    trace: TraceData
    trace_manifest: TraceBundleManifest | None = None
    artifacts: list[ArtifactManifest] = Field(default_factory=list)
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    warnings: list[str] = Field(default_factory=list)
