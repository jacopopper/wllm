from __future__ import annotations

from typing import Any, Protocol

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from schemas.extraction import ExtractRequest
from schemas.openai import ChatCompletionRequest, CompletionRequest
from schemas.traces import TraceEnvelope


class InferenceRuntime(Protocol):
    def list_models(self) -> dict[str, Any]:
        ...

    def capabilities(self) -> Any:
        ...

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        ...

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        ...

    def generate_extract(
        self,
        request: ExtractRequest,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
    ) -> TraceEnvelope:
        ...
