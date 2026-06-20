"""Comprehensive OpenAI-compatible API compliance tests for all 6 endpoints.

These tests exercise the full FastAPI stack (middleware, exception handlers,
routes, schemas) through ``TestClient`` so they validate the wire-level
behavior described in the validation contract (VAL-API-001 .. VAL-API-024).

They complement ``tests/unit/test_app.py`` (which calls route functions
directly) by asserting HTTP status codes, OpenAI-style error envelopes, and
response field shapes end-to-end.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from runtime.capabilities import default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator
from runtime.vllm_compat import VLLMImports
from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
from schemas.extraction import ExtractRequest
from schemas.openai import ChatCompletionRequest, CompletionRequest
from schemas.traces import TokenTrace, TraceData, TraceEnvelope, TraceMetadata
from server.app import create_app
from server.errors import RuntimeUnavailableError

try:  # TraceBundleManifest is needed for /v1/traces persisted responses.
    from schemas.artifacts import TraceBundleManifest
except Exception:  # pragma: no cover - import is always available in-tree
    TraceBundleManifest = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SamplingParams:
    """Fake vLLM SamplingParams that records every accepted kwarg."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeCompletion:
    def __init__(self, text: str, token_ids: list[int], finish_reason: str, logprobs: Any | None = None) -> None:
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.logprobs = logprobs


class _FakeOutput:
    def __init__(self, prompt_token_ids: list[int], completions: list[_FakeCompletion]) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = completions


class _FakeTokenizer:
    """Fake tokenizer exposing only the surface used by ``_render_chat``."""

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: Any) -> str:
        return "rendered:" + "|".join(m.get("role", "") + ":" + m.get("content", "") for m in messages)


class _FakeLLM:
    def __init__(self, outputs: list[_FakeOutput]) -> None:
        self._outputs = outputs
        self._tokenizer = _FakeTokenizer()
        self.generate_calls: list[tuple[list[str], _SamplingParams]] = []

    def get_tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer

    def generate(self, prompts: list[str], sampling: _SamplingParams) -> list[_FakeOutput]:
        self.generate_calls.append((prompts, sampling))
        return self._outputs


def _vllm_imports() -> VLLMImports:
    return VLLMImports(module=object(), version="0.10.2", LLM=object(), SamplingParams=_SamplingParams)


def make_vllm_runtime(
    outputs: list[_FakeOutput],
    *,
    served_model_name: str | None = None,
) -> VLLMRuntime:
    """Build a real ``VLLMRuntime`` wired to fake vLLM objects.

    The runtime's ``_ensure_loaded`` early-returns once ``_llm`` is set, so we
    inject a fake LLM and fake imports to exercise the production response
    shaping code (``_generate_openai_response``) without requiring GPU/vLLM.
    """
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake-model", served_model_name=served_model_name))
    runtime._imports = _vllm_imports()  # type: ignore[attr-defined]
    runtime._llm = _FakeLLM(outputs)  # type: ignore[attr-defined]
    return runtime


class ComplianceRuntime:
    """Fake runtime that runs the real extraction preflight and builds traces.

    Used for endpoint-level tests that need 413/501 errors to surface through
    the real ``ExtractionOrchestrator.preflight`` path while keeping generation
    deterministic and GPU-free.
    """

    def __init__(self, served_model_name: str | None = None) -> None:
        self.model = "fake-model"
        self.served_model_name = served_model_name or self.model
        self.last_chat_request: ChatCompletionRequest | None = None
        self.last_completion_request: CompletionRequest | None = None

    def capabilities(self) -> Any:
        return default_vllm_capabilities(self.model, "0.10.2")

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "wllm",
                }
            ],
        }

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        self.last_chat_request = request
        return {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        self.last_completion_request = request
        return {
            "id": f"cmpl_{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def generate_extract(
        self,
        request: ExtractRequest,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
    ) -> TraceEnvelope:
        # Run the real preflight so 413/501 paths are exercised end-to-end.
        ExtractionOrchestrator(self.capabilities()).preflight(request, limits, topology=None)
        manifest = None
        if persist and TraceBundleManifest is not None:
            manifest = TraceBundleManifest(
                manifest_id=f"manifest_{uuid.uuid4().hex}",
                schema_version="wllm.trace.v1",
                path=f"traces/trace_{uuid.uuid4().hex}.json",
                byte_size=128,
                sha256="0" * 64,
                created=int(time.time()),
                trace_id=f"trace_{uuid.uuid4().hex}",
            )
        return TraceEnvelope(
            id=f"trace_{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.served_model_name,
            generation={
                "id": "cmpl_fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
            trace=TraceData(
                tokens=TokenTrace(token_ids=[1, 2, 3], tokens=["a", "b", "c"]),
                spans={"prompt": (0, 2), "generated": (2, 3)},
            ),
            trace_manifest=manifest,
            metadata=TraceMetadata(capabilities=self.capabilities().as_metadata()),
        )


class UnavailableRuntime(ComplianceRuntime):
    """Runtime whose generation methods report the runtime as unavailable (503)."""

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        raise RuntimeUnavailableError("vLLM is not available.", code="vllm_initialization_failed")

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        raise RuntimeUnavailableError("vLLM is not available.", code="vllm_initialization_failed")


class CrashingRuntime(ComplianceRuntime):
    """Runtime whose generation methods raise an unexpected exception (500)."""

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        raise RuntimeError("internal boom")

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        raise RuntimeError("internal boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(runtime: Any, *, api_key: str | None = None, limits: ResourceLimits | None = None) -> TestClient:
    app = create_app(runtime=runtime, api_key=api_key, limits=limits or ResourceLimits())
    # raise_server_exceptions=False lets the registered 500 handler return its
    # JSON envelope instead of re-raising inside the test transport.
    return TestClient(app, raise_server_exceptions=False)


def _assert_error_envelope(body: dict[str, Any]) -> dict[str, Any]:
    """Assert the OpenAI-style ``{"error": {...}}`` envelope with all 5 fields."""
    assert "error" in body, f"missing top-level 'error' envelope in {body}"
    error = body["error"]
    for field in ("message", "type", "param", "code", "details"):
        assert field in error, f"error envelope missing '{field}' in {error}"
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["type"], str) and error["type"]
    assert error["param"] is None or isinstance(error["param"], str)
    assert isinstance(error["code"], str) and error["code"]
    return error


_CHAT_PAYLOAD = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
_COMPLETION_PAYLOAD = {"model": "fake-model", "prompt": "hi", "max_tokens": 4}


# ---------------------------------------------------------------------------
# VAL-API-001: GET /v1/models list shape
# ---------------------------------------------------------------------------


def test_models_endpoint_returns_list_shape() -> None:
    # covers VAL-API-001
    with _client(ComplianceRuntime()) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and body["data"]
    for entry in body["data"]:
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["created"], int)
        assert isinstance(entry["owned_by"], str) and entry["owned_by"]


# ---------------------------------------------------------------------------
# VAL-API-002 / VAL-API-003: chat & completion response shapes (HTTP level)
# ---------------------------------------------------------------------------


def test_chat_completions_endpoint_response_shape() -> None:
    # covers VAL-API-002
    with _client(ComplianceRuntime()) as client:
        response = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["id"], str) and body["id"]
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert isinstance(body["model"], str)
    assert isinstance(body["choices"], list) and body["choices"]
    choice = body["choices"][0]
    assert isinstance(choice["index"], int)
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    usage = body["usage"]
    assert isinstance(usage["prompt_tokens"], int)
    assert isinstance(usage["completion_tokens"], int)
    assert isinstance(usage["total_tokens"], int)


def test_completions_endpoint_response_shape() -> None:
    # covers VAL-API-003
    with _client(ComplianceRuntime()) as client:
        response = client.post("/v1/completions", json=_COMPLETION_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["id"], str) and body["id"]
    assert body["object"] == "text_completion"
    assert isinstance(body["created"], int)
    assert isinstance(body["model"], str)
    assert isinstance(body["choices"], list) and body["choices"]
    choice = body["choices"][0]
    assert isinstance(choice["index"], int)
    assert isinstance(choice["text"], str)
    usage = body["usage"]
    assert isinstance(usage["prompt_tokens"], int)
    assert isinstance(usage["completion_tokens"], int)
    assert isinstance(usage["total_tokens"], int)


# ---------------------------------------------------------------------------
# VAL-API-004 / VAL-API-005: streaming rejected with 422
# ---------------------------------------------------------------------------


def test_chat_streaming_rejected_with_422_envelope() -> None:
    # covers VAL-API-004
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert response.status_code == 422
    error = _assert_error_envelope(response.json())
    assert error["code"] == "streaming_not_implemented"
    assert error["param"] == "stream"
    assert error["type"] == "invalid_request_error"


def test_extract_streaming_rejected_with_422_envelope() -> None:
    # covers VAL-API-005
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={"model": "fake-model", "prompt": "hi", "extract": {"tokens": True}, "stream": True},
        )
    assert response.status_code == 422
    error = _assert_error_envelope(response.json())
    assert error["code"] == "streaming_not_implemented"
    assert error["param"] == "stream"


def test_traces_streaming_rejected_with_422_envelope() -> None:
    # bundled: streaming rejection for /v1/traces (same code path)
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/traces",
            json={"model": "fake-model", "prompt": "hi", "extract": {"tokens": True}, "stream": True},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "streaming_not_implemented"


def test_completions_streaming_rejected_with_422_envelope() -> None:
    # bundled: streaming rejection for /v1/completions
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "hi", "stream": True},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "streaming_not_implemented"


# ---------------------------------------------------------------------------
# VAL-API-006 / VAL-API-007: served-model-name propagation
# ---------------------------------------------------------------------------


def test_served_model_name_reflected_in_models_endpoint() -> None:
    # covers VAL-API-006
    with _client(ComplianceRuntime(served_model_name="my-custom-name")) as client:
        response = client.get("/v1/models")
    body = response.json()
    ids = [entry["id"] for entry in body["data"]]
    assert "my-custom-name" in ids
    entry = next(entry for entry in body["data"] if entry["id"] == "my-custom-name")
    assert isinstance(entry["created"], int)
    assert isinstance(entry["owned_by"], str) and entry["owned_by"]


def test_served_model_name_reflected_in_generation_responses() -> None:
    # covers VAL-API-007
    with _client(ComplianceRuntime(served_model_name="my-custom-name")) as client:
        chat = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
        completion = client.post("/v1/completions", json=_COMPLETION_PAYLOAD)
    assert chat.json()["model"] == "my-custom-name"
    assert completion.json()["model"] == "my-custom-name"


# ---------------------------------------------------------------------------
# VAL-API-008 / VAL-API-009 / VAL-API-010: API key middleware
# ---------------------------------------------------------------------------


def test_api_key_middleware_blocks_unauthorized_with_401() -> None:
    # covers VAL-API-008
    with _client(ComplianceRuntime(), api_key="my-secret") as client:
        response = client.get("/v1/models")
    assert response.status_code == 401
    error = _assert_error_envelope(response.json())
    assert error["code"] == "invalid_api_key"
    assert error["type"] == "authentication_error"


def test_api_key_middleware_accepts_bearer_and_bare_and_rejects_wrong() -> None:
    # covers VAL-API-009
    with _client(ComplianceRuntime(), api_key="my-secret") as client:
        assert client.get("/v1/models", headers={"Authorization": "Bearer my-secret"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "my-secret"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "wrong-key"}).status_code == 401


def test_no_api_key_allows_all_requests() -> None:
    # covers VAL-API-010
    with _client(ComplianceRuntime()) as client:
        models = client.get("/v1/models")
        completion = client.post("/v1/completions", json=_COMPLETION_PAYLOAD)
    assert models.status_code == 200
    assert completion.status_code == 200


# ---------------------------------------------------------------------------
# VAL-API-011: unexpected errors -> 500 OpenAI envelope
# ---------------------------------------------------------------------------


def test_unexpected_error_produces_500_openai_envelope() -> None:
    # covers VAL-API-011
    with _client(CrashingRuntime()) as client:
        response = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
    assert response.status_code == 500
    error = _assert_error_envelope(response.json())
    assert error["code"] == "internal_error"
    assert error["type"] == "internal_server_error"
    # Raw traceback must not leak.
    assert "boom" not in error["message"]
    assert "RuntimeError" not in response.text


# ---------------------------------------------------------------------------
# VAL-API-012: validation errors -> 422 OpenAI envelope (schema_validation_failed)
# ---------------------------------------------------------------------------


def test_validation_error_produces_422_openai_envelope() -> None:
    # covers VAL-API-012
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt": "hi",
                "extract": {"tokens": True},
            },
        )
    assert response.status_code == 422
    error = _assert_error_envelope(response.json())
    assert error["code"] == "schema_validation_failed"
    assert error["type"] == "invalid_request_error"
    errors = error["details"]["errors"]
    assert isinstance(errors, list) and errors
    # Raw input must not be leaked.
    assert all("input" not in entry for entry in errors)


def test_unknown_request_fields_rejected_with_422() -> None:
    # expectedBehavior: unknown request fields rejected with 422
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "foo": "bar"},
        )
    assert response.status_code == 422
    error = _assert_error_envelope(response.json())
    assert error["code"] == "schema_validation_failed"
    locs = [tuple(entry["loc"]) for entry in error["details"]["errors"]]
    assert any("foo" in loc for loc in locs)


def test_non_json_body_returns_structured_error() -> None:
    # expectedBehavior: non-JSON body returns structured error
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/chat/completions",
            content="this is not json",
            headers={"Content-Type": "text/plain"},
        )
    assert response.status_code == 422
    error = _assert_error_envelope(response.json())
    assert error["code"] == "schema_validation_failed"


# ---------------------------------------------------------------------------
# VAL-API-013: /v1/extraction-schema versioned schema with limits + capabilities
# ---------------------------------------------------------------------------


def test_extraction_schema_endpoint_returns_versioned_payload() -> None:
    # covers VAL-API-013
    with _client(ComplianceRuntime()) as client:
        response = client.get("/v1/extraction-schema")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "wllm.extraction.v1"
    assert body["request_schema"]["title"] == "ExtractRequest"
    assert "extract" in body["request_schema"]["properties"]
    assert "selectors" in body
    limits = body["limits"]
    for field in (
        "max_top_k",
        "max_selected_layers",
        "max_selected_heads",
        "max_selected_positions",
        "max_inline_tensor_bytes",
        "max_total_captured_tensor_bytes",
        "max_artifact_bytes",
        "large_extraction_enabled",
    ):
        assert field in limits, f"limits missing {field}"
    capabilities = body["capabilities"]
    for field in (
        "token_ids",
        "hidden_states",
        "attentions",
        "npz_artifacts",
        "pt_artifacts",
    ):
        assert field in capabilities, f"capabilities missing {field}"


# ---------------------------------------------------------------------------
# VAL-API-014: /v1/extract returns wllm.trace.v1 with trace_manifest absent
# ---------------------------------------------------------------------------


def test_extract_endpoint_returns_trace_schema_without_manifest() -> None:
    # covers VAL-API-014
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={"model": "fake-model", "prompt": "hi", "max_tokens": 2, "extract": {"tokens": True}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "wllm.trace.v1"
    assert body["object"] == "wllm.trace"
    assert isinstance(body["id"], str) and body["id"]
    assert isinstance(body["created"], int)
    assert isinstance(body["model"], str)
    assert "generation" in body and "trace" in body and "metadata" in body and "warnings" in body
    # persist=False -> no trace manifest and no artifacts.
    assert body.get("trace_manifest") is None
    assert body.get("artifacts", []) == []


# ---------------------------------------------------------------------------
# VAL-API-015: /v1/traces returns wllm.trace.v1 with non-null trace_manifest
# ---------------------------------------------------------------------------


def test_traces_endpoint_returns_trace_with_manifest() -> None:
    # covers VAL-API-015
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/traces",
            json={"model": "fake-model", "prompt": "hi", "max_tokens": 2, "extract": {"tokens": True}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "wllm.trace.v1"
    manifest = body.get("trace_manifest")
    assert manifest is not None
    assert manifest["object"] == "wllm.trace_manifest"
    assert isinstance(manifest["manifest_id"], str) and manifest["manifest_id"]
    assert isinstance(manifest["trace_id"], str) and manifest["trace_id"]
    assert isinstance(manifest["sha256"], str) and len(manifest["sha256"]) == 64
    assert isinstance(manifest["byte_size"], int) and manifest["byte_size"] >= 0
    assert isinstance(manifest["created"], int)


# ---------------------------------------------------------------------------
# VAL-API-016: chat accepts multi-role messages; empty messages rejected
# ---------------------------------------------------------------------------


def test_chat_accepts_multi_role_messages_and_rejects_empty() -> None:
    # covers VAL-API-016
    runtime = ComplianceRuntime()
    with _client(runtime) as client:
        ok = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "bye"},
                ],
                "max_tokens": 4,
            },
        )
        empty = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [], "max_tokens": 4},
        )
    assert ok.status_code == 200
    assert runtime.last_chat_request is not None
    assert [m.role for m in runtime.last_chat_request.messages] == ["system", "user", "assistant", "user"]
    assert empty.status_code == 422
    _assert_error_envelope(empty.json())


# ---------------------------------------------------------------------------
# VAL-API-017: completion accepts raw prompt string and list; empty/missing rejected
# ---------------------------------------------------------------------------


def test_completion_accepts_string_and_list_and_rejects_empty_missing() -> None:
    # covers VAL-API-017
    with _client(ComplianceRuntime()) as client:
        single = client.post("/v1/completions", json={"model": "fake-model", "prompt": "hello", "max_tokens": 2})
        many = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": ["text1", "text2"], "max_tokens": 2},
        )
        empty = client.post("/v1/completions", json={"model": "fake-model", "prompt": "", "max_tokens": 2})
        missing = client.post("/v1/completions", json={"model": "fake-model", "max_tokens": 2})
    assert single.status_code == 200
    assert many.status_code == 200
    assert empty.status_code == 422
    _assert_error_envelope(empty.json())
    assert missing.status_code == 422
    _assert_error_envelope(missing.json())


# ---------------------------------------------------------------------------
# VAL-API-018: error envelope shape consistent across all status codes
# ---------------------------------------------------------------------------


def test_error_envelope_shape_consistent_across_status_codes() -> None:
    # covers VAL-API-018
    cases: list[tuple[int, dict[str, Any]]] = []

    # 401 via missing auth
    with _client(ComplianceRuntime(), api_key="secret") as client:
        cases.append((client.get("/v1/models").status_code, client.get("/v1/models").json()))

    # 413 via exceeding max_top_k
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={
                "model": "fake-model",
                "prompt": "hi",
                "extract": {"logprobs": {"top_k": 999}},
            },
        )
    cases.append((response.status_code, response.json()))

    # 422 via streaming + via bad schema
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    cases.append((response.status_code, response.json()))

    # 500 via internal crash
    with _client(CrashingRuntime()) as client:
        response = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
    cases.append((response.status_code, response.json()))

    # 501 via unsupported attention extraction
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={
                "model": "fake-model",
                "prompt": "hi",
                "extract": {
                    "attentions": [
                        {
                            "layers": "middle",
                            "heads": "all",
                            "query_positions": "generated",
                            "key_positions": "previous_token",
                        }
                    ]
                },
            },
        )
    cases.append((response.status_code, response.json()))

    # 503 via runtime unavailable
    with _client(UnavailableRuntime()) as client:
        response = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
    cases.append((response.status_code, response.json()))

    expected = {401, 413, 422, 500, 501, 503}
    observed = {status for status, _ in cases}
    assert observed == expected, f"missing status codes: {expected - observed}, extra: {observed - expected}"

    for status, body in cases:
        error = _assert_error_envelope(body)
        assert error["type"] in {
            "authentication_error",
            "resource_limit_exceeded",
            "invalid_request_error",
            "internal_server_error",
            "unsupported_extraction",
            "runtime_unavailable",
        }, f"unexpected error type {error['type']} for status {status}"
        assert isinstance(error["code"], str) and error["code"]
        assert error["param"] is None or isinstance(error["param"], str)


# ---------------------------------------------------------------------------
# VAL-API-019: usage.total_tokens == prompt_tokens + completion_tokens
# ---------------------------------------------------------------------------


def test_usage_total_equals_prompt_plus_completion() -> None:
    # covers VAL-API-019
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[10, 11, 12], completions=[_FakeCompletion("a b c", [20, 21, 22], "stop")])]
    )
    request = ChatCompletionRequest.model_validate(
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    )
    body = runtime.generate_chat(request)
    usage = body["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] == 6

    # Vary token counts through completions endpoint with multiple completions.
    runtime2 = make_vllm_runtime(
        [
            _FakeOutput(
                prompt_token_ids=[1, 2],
                completions=[_FakeCompletion("x", [3], "stop"), _FakeCompletion("y z", [4, 5], "length")],
            )
        ]
    )
    request2 = CompletionRequest.model_validate({"model": "fake-model", "prompt": "hi", "max_tokens": 2, "n": 2})
    body2 = runtime2.generate_completion(request2)
    usage2 = body2["usage"]
    assert usage2["total_tokens"] == usage2["prompt_tokens"] + usage2["completion_tokens"]
    assert usage2["prompt_tokens"] == 2
    assert usage2["completion_tokens"] == 3


# ---------------------------------------------------------------------------
# VAL-API-020 / VAL-API-021: finish_reason semantics
# ---------------------------------------------------------------------------


def test_finish_reason_length_when_max_tokens_exhausted() -> None:
    # covers VAL-API-020
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[1, 2], completions=[_FakeCompletion("only one", [3], "length")])]
    )
    request = ChatCompletionRequest.model_validate(
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    )
    body = runtime.generate_chat(request)
    assert body["choices"][0]["finish_reason"] == "length"


def test_finish_reason_stop_when_eos_generated() -> None:
    # covers VAL-API-021
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[1, 2], completions=[_FakeCompletion("done", [3], "stop")])]
    )
    request = ChatCompletionRequest.model_validate(
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    )
    body = runtime.generate_chat(request)
    assert body["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# VAL-API-022: resource limit exceeded -> 413
# ---------------------------------------------------------------------------


def test_resource_limit_exceeded_returns_413() -> None:
    # covers VAL-API-022
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={"model": "fake-model", "prompt": "hi", "extract": {"logprobs": {"top_k": 999}}},
        )
    assert response.status_code == 413
    error = _assert_error_envelope(response.json())
    assert error["type"] == "resource_limit_exceeded"
    assert error["code"] == "extraction_limit_exceeded"


# ---------------------------------------------------------------------------
# VAL-API-023: unsupported extraction -> 501
# ---------------------------------------------------------------------------


def test_unsupported_extraction_returns_501() -> None:
    # covers VAL-API-023
    with _client(ComplianceRuntime()) as client:
        response = client.post(
            "/v1/extract",
            json={
                "model": "fake-model",
                "prompt": "hi",
                "extract": {
                    "attentions": [
                        {
                            "layers": "middle",
                            "heads": "all",
                            "query_positions": "generated",
                            "key_positions": "previous_token",
                        }
                    ]
                },
            },
        )
    assert response.status_code == 501
    error = _assert_error_envelope(response.json())
    assert error["type"] == "unsupported_extraction"
    assert error["code"] == "attention_weights_unavailable"


def test_runtime_unavailable_returns_503() -> None:
    # bundled: covers VAL-API-018 503 leg explicitly + expectedBehavior 503
    with _client(UnavailableRuntime()) as client:
        response = client.post("/v1/chat/completions", json=_CHAT_PAYLOAD)
    assert response.status_code == 503
    error = _assert_error_envelope(response.json())
    assert error["type"] == "runtime_unavailable"
    assert error["code"] == "vllm_initialization_failed"


# ---------------------------------------------------------------------------
# VAL-API-024: omitted max_tokens defaults to 16
# ---------------------------------------------------------------------------


def test_max_tokens_default_16_honored_when_omitted() -> None:
    # covers VAL-API-024
    # Schema-level default.
    parsed = ChatCompletionRequest.model_validate(
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert parsed.max_tokens == 16

    # The default is forwarded to the runtime as max_tokens=16.
    runtime = ComplianceRuntime()
    with _client(runtime) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert runtime.last_chat_request is not None
    assert runtime.last_chat_request.max_tokens == 16

    # Completion endpoint honors the same default.
    with _client(runtime) as client:
        client.post("/v1/completions", json={"model": "fake-model", "prompt": "hi"})
    assert runtime.last_completion_request is not None
    assert runtime.last_completion_request.max_tokens == 16


# ---------------------------------------------------------------------------
# expectedBehavior: wrong HTTP method returns 405
# ---------------------------------------------------------------------------


def test_wrong_http_method_returns_405() -> None:
    # expectedBehavior: wrong HTTP method returns 405
    with _client(ComplianceRuntime()) as client:
        post_models = client.post("/v1/models")
        get_extract = client.get("/v1/extract")
        get_chat = client.get("/v1/chat/completions")
        post_schema = client.post("/v1/extraction-schema")
    for response in (post_models, get_extract, get_chat, post_schema):
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# expectedBehavior: sampling parameter propagation (temperature, seed, stop, frequency_penalty)
# ---------------------------------------------------------------------------


def test_sampling_parameters_propagate_to_vllm_sampling_params() -> None:
    # expectedBehavior: sampling parameter propagation through the real runtime.
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[1], completions=[_FakeCompletion("ok", [2], "stop")])]
    )
    fake_llm = runtime._llm  # type: ignore[attr-defined]
    request = ChatCompletionRequest.model_validate(
        {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.7,
            "seed": 42,
            "stop": ["end"],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "top_p": 0.9,
        }
    )
    runtime.generate_chat(request)
    assert len(fake_llm.generate_calls) == 1
    sampling = fake_llm.generate_calls[0][1]
    assert sampling.kwargs["max_tokens"] == 4
    assert sampling.kwargs["temperature"] == 0.7
    assert sampling.kwargs["seed"] == 42
    assert sampling.kwargs["stop"] == ["end"]
    assert sampling.kwargs["frequency_penalty"] == 0.5
    assert sampling.kwargs["presence_penalty"] == 0.3
    assert sampling.kwargs["top_p"] == 0.9


def test_sampling_parameters_forwarded_to_runtime_via_http() -> None:
    # expectedBehavior (HTTP level): route parses and forwards sampling params.
    runtime = ComplianceRuntime()
    with _client(runtime) as client:
        client.post(
            "/v1/completions",
            json={
                "model": "fake-model",
                "prompt": "hi",
                "max_tokens": 4,
                "temperature": 0.5,
                "seed": 7,
                "stop": ["done"],
                "frequency_penalty": 0.25,
            },
        )
    request = runtime.last_completion_request
    assert request is not None
    assert request.temperature == 0.5
    assert request.seed == 7
    assert request.stop == ["done"]
    assert request.frequency_penalty == 0.25


# ---------------------------------------------------------------------------
# expectedBehavior: choices[].index correctness + id field format
# ---------------------------------------------------------------------------


def test_choices_index_correctness_single_prompt_multiple_choices() -> None:
    # expectedBehavior: choices[].index correctness (n=2, single prompt)
    runtime = make_vllm_runtime(
        [
            _FakeOutput(
                prompt_token_ids=[1, 2],
                completions=[
                    _FakeCompletion("a", [3], "stop"),
                    _FakeCompletion("b", [4], "stop"),
                ],
            )
        ]
    )
    request = ChatCompletionRequest.model_validate(
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": 2}
    )
    body = runtime.generate_chat(request)
    assert [choice["index"] for choice in body["choices"]] == [0, 1]


def test_choices_index_correctness_multiple_prompts() -> None:
    # expectedBehavior: choices[].index correctness (2 prompts, n=1)
    runtime = make_vllm_runtime(
        [
            _FakeOutput(prompt_token_ids=[1], completions=[_FakeCompletion("a", [2], "stop")]),
            _FakeOutput(prompt_token_ids=[3], completions=[_FakeCompletion("b", [4], "stop")]),
        ]
    )
    request = CompletionRequest.model_validate({"model": "fake-model", "prompt": ["p1", "p2"], "max_tokens": 1})
    body = runtime.generate_completion(request)
    assert [choice["index"] for choice in body["choices"]] == [0, 1]


def test_choices_index_correctness_multiple_prompts_and_choices() -> None:
    # expectedBehavior: choices[].index correctness (2 prompts x n=2 -> 0,1,2,3)
    runtime = make_vllm_runtime(
        [
            _FakeOutput(
                prompt_token_ids=[1],
                completions=[_FakeCompletion("a", [2], "stop"), _FakeCompletion("b", [3], "stop")],
            ),
            _FakeOutput(
                prompt_token_ids=[4],
                completions=[_FakeCompletion("c", [5], "stop"), _FakeCompletion("d", [6], "stop")],
            ),
        ]
    )
    request = CompletionRequest.model_validate(
        {"model": "fake-model", "prompt": ["p1", "p2"], "max_tokens": 1, "n": 2}
    )
    body = runtime.generate_completion(request)
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2, 3]


def test_id_field_format_chat_and_completion() -> None:
    # expectedBehavior: id field format verification
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[1], completions=[_FakeCompletion("ok", [2], "stop")])]
    )
    chat = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        )
    )
    completion = runtime.generate_completion(
        CompletionRequest.model_validate({"model": "fake-model", "prompt": "hi", "max_tokens": 1})
    )
    chat_id = chat["id"]
    completion_id = completion["id"]
    assert chat_id.startswith("chatcmpl_")
    assert completion_id.startswith("cmpl_")
    # The suffix is uuid4().hex (32 hex chars).
    chat_suffix = chat_id[len("chatcmpl_"):]
    completion_suffix = completion_id[len("cmpl_"):]
    assert len(chat_suffix) == 32
    assert len(completion_suffix) == 32
    int(chat_suffix, 16)  # raises if not hex
    int(completion_suffix, 16)
    # Identifiers must be unique across calls.
    assert chat_id != completion_id


def test_created_timestamp_is_unix_integer() -> None:
    # bundled: covers response field 'created' timestamp detail
    runtime = make_vllm_runtime(
        [_FakeOutput(prompt_token_ids=[1], completions=[_FakeCompletion("ok", [2], "stop")])]
    )
    before = int(time.time())
    body = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        )
    )
    after = int(time.time())
    assert isinstance(body["created"], int)
    assert before <= body["created"] <= after


# ---------------------------------------------------------------------------
# Sanity: extraction schema endpoint reachable without auth when no key set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/v1/models", "/v1/extraction-schema"])
def test_read_endpoints_reachable_without_auth(endpoint: str) -> None:
    # bundled: reinforces VAL-API-010 for read endpoints
    with _client(ComplianceRuntime()) as client:
        response = client.get(endpoint)
    assert response.status_code == 200
