from __future__ import annotations

import asyncio
import json

from starlette.exceptions import HTTPException as StarletteHTTPException

from server.app import create_app
from server.errors import http_exception_handler, unexpected_exception_handler
from tests.unit.test_app import FakeRuntime


def test_unexpected_exception_handler_uses_openai_style_500() -> None:
    response = asyncio.run(unexpected_exception_handler(None, RuntimeError("boom")))
    body = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 500
    assert body == {
        "error": {
            "message": "Unexpected internal server error.",
            "type": "internal_server_error",
            "status": 500,
            "param": None,
            "code": "internal_error",
            "details": {},
        }
    }


def test_app_registers_generic_exception_handler() -> None:
    app = create_app(runtime=FakeRuntime())
    assert app.exception_handlers[Exception] is unexpected_exception_handler
    assert app.exception_handlers[StarletteHTTPException] is http_exception_handler


def test_validation_error_response_sanitizes_non_json_context() -> None:
    from fastapi.testclient import TestClient

    app = create_app(runtime=FakeRuntime())
    client = TestClient(app)

    response = client.post("/v1/extract", json={"model": "fake-model", "extract": {"tokens": True}})

    assert response.status_code == 422
    body = response.json()
    ctx = body["error"]["details"]["errors"][0]["ctx"]
    assert isinstance(ctx["error"], str)
    assert "exactly one of messages or prompt" in ctx["error"]
