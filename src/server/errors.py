from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class WLLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str,
        code: str,
        param: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.details = details or {}


class InvalidRequestError(WLLMError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        param: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=422,
            error_type="invalid_request_error",
            code=code,
            param=param,
            details=details,
        )


class ResourceLimitError(WLLMError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "extraction_limit_exceeded",
        param: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=413,
            error_type="resource_limit_exceeded",
            code=code,
            param=param,
            details=details,
        )


class UnsupportedExtractionError(WLLMError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        param: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=501,
            error_type="unsupported_extraction",
            code=code,
            param=param,
            details=details,
        )


class RuntimeUnavailableError(WLLMError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "runtime_unavailable",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=503,
            error_type="runtime_unavailable",
            code=code,
            details=details,
        )


class InternalServerError(WLLMError):
    def __init__(self) -> None:
        super().__init__(
            "Unexpected internal server error.",
            status_code=500,
            error_type="internal_server_error",
            code="internal_error",
        )


def error_envelope(exc: WLLMError) -> dict[str, Any]:
    return {
        "error": {
            "message": exc.message,
            "type": exc.error_type,
            "param": exc.param,
            "code": exc.code,
            "details": exc.details,
        }
    }


async def wllm_exception_handler(_request: Request, exc: WLLMError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=error_envelope(exc))


async def request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for error in exc.errors():
        cleaned = {key: _json_safe(value) for key, value in error.items() if key != "input"}
        details.append(cleaned)
    api_error = InvalidRequestError(
        "Request validation failed.",
        code="schema_validation_failed",
        details={"errors": details},
    )
    return JSONResponse(status_code=api_error.status_code, content=error_envelope(api_error))


async def unexpected_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    api_error = InternalServerError()
    return JSONResponse(status_code=api_error.status_code, content=error_envelope(api_error))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
