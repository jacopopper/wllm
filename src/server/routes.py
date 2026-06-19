from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from extractors.planning import ResourceLimits
from schemas.extraction import ExtractionSchemaResponse, ExtractRequest, TraceRequest, extraction_schema_payload
from schemas.openai import ChatCompletionRequest, CompletionRequest
from server.errors import InvalidRequestError

router = APIRouter()


def runtime_from_request(request: Request) -> Any:
    return request.app.state.runtime


def limits_from_request(request: Request) -> ResourceLimits:
    return request.app.state.limits


def artifact_store_from_request(request: Request) -> Any:
    return request.app.state.artifact_store


@router.get("/v1/models")
def list_models(runtime: Any = Depends(runtime_from_request)) -> dict[str, Any]:
    return runtime.list_models()


@router.post("/v1/chat/completions")
def chat_completions(
    body: ChatCompletionRequest,
    runtime: Any = Depends(runtime_from_request),
) -> dict[str, Any]:
    if body.stream:
        raise InvalidRequestError(
            "Streaming chat completions are not implemented.",
            code="streaming_not_implemented",
            param="stream",
        )
    return runtime.generate_chat(body)


@router.post("/v1/completions")
def completions(
    body: CompletionRequest,
    runtime: Any = Depends(runtime_from_request),
) -> dict[str, Any]:
    if body.stream:
        raise InvalidRequestError(
            "Streaming completions are not implemented.",
            code="streaming_not_implemented",
            param="stream",
        )
    return runtime.generate_completion(body)


@router.post("/v1/extract")
def extract(
    body: ExtractRequest,
    runtime: Any = Depends(runtime_from_request),
    limits: ResourceLimits = Depends(limits_from_request),
    artifact_store: Any = Depends(artifact_store_from_request),
) -> dict[str, Any]:
    if body.stream:
        raise InvalidRequestError(
            "Streaming extraction is not implemented.",
            code="streaming_not_implemented",
            param="stream",
        )
    trace = runtime.generate_extract(body, limits=limits, artifact_store=artifact_store, persist=False)
    return trace.model_dump(mode="json", exclude_none=True)


@router.post("/v1/traces")
def traces(
    body: TraceRequest,
    runtime: Any = Depends(runtime_from_request),
    limits: ResourceLimits = Depends(limits_from_request),
    artifact_store: Any = Depends(artifact_store_from_request),
) -> dict[str, Any]:
    if body.stream:
        raise InvalidRequestError(
            "Streaming trace extraction is not implemented.",
            code="streaming_not_implemented",
            param="stream",
        )
    trace = runtime.generate_extract(body, limits=limits, artifact_store=artifact_store, persist=True)
    return trace.model_dump(mode="json", exclude_none=True)


@router.get("/v1/extraction-schema")
def extraction_schema(
    runtime: Any = Depends(runtime_from_request),
    limits: ResourceLimits = Depends(limits_from_request),
) -> ExtractionSchemaResponse:
    capabilities = runtime.capabilities()
    return extraction_schema_payload(limits=limits, capabilities=capabilities)
