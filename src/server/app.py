from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from runtime.base import InferenceRuntime
from server.errors import WLLMError, request_validation_handler, unexpected_exception_handler, wllm_exception_handler
from server.routes import router


def create_app(
    runtime: InferenceRuntime,
    artifact_store: ArtifactStore | None = None,
    limits: ResourceLimits | None = None,
    api_key: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="wllm",
        version="0.1.0",
        description="vLLM for AI safety researchers: fast serving plus runtime-selectable model internals.",
    )
    app.state.runtime = runtime
    app.state.artifact_store = artifact_store or ArtifactStore(Path("./wllm-artifacts"))
    app.state.limits = limits or ResourceLimits()
    app.state.api_key = api_key
    if api_key is not None:
        app.middleware("http")(_api_key_middleware)
    app.add_exception_handler(WLLMError, cast(Any, wllm_exception_handler))
    app.add_exception_handler(RequestValidationError, cast(Any, request_validation_handler))
    app.add_exception_handler(Exception, cast(Any, unexpected_exception_handler))
    app.include_router(router)
    return app


async def _api_key_middleware(request: Request, call_next):
    expected = request.app.state.api_key
    if expected is None:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        token = header[7:]
    else:
        token = header
    if token != expected:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Unauthorized: invalid or missing API key.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                    "details": {},
                }
            },
        )
    return await call_next(request)
