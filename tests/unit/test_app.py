from __future__ import annotations

import inspect
import time
import uuid
from typing import Any

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from runtime.capabilities import default_vllm_capabilities
from schemas.extraction import ExtractRequest
from schemas.openai import ChatCompletionRequest, CompletionRequest
from schemas.traces import TokenTrace, TraceData, TraceEnvelope, TraceMetadata
from server.app import create_app
from server.errors import UnsupportedExtractionError, WLLMError
from server.routes import chat_completions, completions, extract, extraction_schema, list_models, traces


class FakeRuntime:
    def __init__(self, served_model_name: str | None = None) -> None:
        self.model = "fake-model"
        self.served_model_name = served_model_name or self.model
        self.extract_calls = 0
        self.chat_calls = 0
        self.completion_calls = 0

    def capabilities(self) -> Any:
        return default_vllm_capabilities(self.model, "fake")

    def list_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": self.served_model_name, "object": "model", "created": 0, "owned_by": "wllm"}]}

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        self.chat_calls += 1
        return {
            "id": "chatcmpl_fake",
            "object": "chat.completion",
            "created": 0,
            "model": self.served_model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        self.completion_calls += 1
        return {
            "id": "cmpl_fake",
            "object": "text_completion",
            "created": 0,
            "model": self.served_model_name,
            "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def generate_extract(
        self,
        request: ExtractRequest,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
    ) -> TraceEnvelope:
        del limits, artifact_store, persist
        self.extract_calls += 1
        if request.extract.attentions:
            raise UnsupportedExtractionError(
                "attention unavailable",
                code="attention_weights_unavailable",
                param="extract.attentions",
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
                logprobs={"generated": [{"top_logprobs": [{"token_id": 3, "logprob": -0.1}]}]},
            ),
            metadata=TraceMetadata(capabilities=self.capabilities().as_metadata()),
        )


def make_app(served_model_name: str | None = None) -> tuple[Any, FakeRuntime]:
    runtime = FakeRuntime(served_model_name=served_model_name)
    return None, runtime


def test_models_response_shape() -> None:
    _app, runtime = make_app()
    body = list_models(runtime=runtime)
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "fake-model"


def test_chat_completion_shape_and_trace_free_path() -> None:
    _app, runtime = make_app()
    body = chat_completions(
        ChatCompletionRequest.model_validate(
            {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        ),
        runtime=runtime,
    )
    assert body["choices"][0]["message"]["content"] == "ok"
    assert runtime.chat_calls == 1
    assert runtime.extract_calls == 0


def test_generation_routes_are_sync_for_fastapi_threadpool() -> None:
    assert not inspect.iscoroutinefunction(chat_completions)
    assert not inspect.iscoroutinefunction(completions)
    assert not inspect.iscoroutinefunction(extract)
    assert not inspect.iscoroutinefunction(traces)


def test_completion_shape() -> None:
    _app, runtime = make_app()
    body = completions(CompletionRequest.model_validate({"model": "fake-model", "prompt": "hi", "max_tokens": 4}), runtime=runtime)
    assert body["choices"][0]["text"] == "ok"
    assert body["usage"]["total_tokens"] == 2


def test_streaming_rejected_explicitly() -> None:
    _app, runtime = make_app()
    try:
        chat_completions(
            ChatCompletionRequest.model_validate(
                {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            ),
            runtime=runtime,
        )
    except WLLMError as exc:
        assert exc.status_code == 422
        assert exc.code == "streaming_not_implemented"
    else:
        raise AssertionError("streaming request should fail")


def test_extraction_streaming_rejected_explicitly() -> None:
    _app, runtime = make_app()
    request = ExtractRequest.model_validate(
        {"model": "fake-model", "prompt": "hi", "stream": True, "extract": {"tokens": True}}
    )

    try:
        extract(
            request,
            runtime=runtime,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
        )
    except WLLMError as exc:
        assert exc.status_code == 422
        assert exc.code == "streaming_not_implemented"
        assert exc.param == "stream"
    else:
        raise AssertionError("streaming extraction request should fail")


def test_extract_trace_shape() -> None:
    _app, runtime = make_app()
    body = extract(
        ExtractRequest.model_validate(
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extract": {"tokens": True, "logprobs": {"top_k": 1}},
            }
        ),
        runtime=runtime,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
    )
    assert body["schema_version"] == "wllm.trace.v1"
    assert body["object"] == "wllm.trace"
    assert body["trace"]["spans"]["prompt"] == [0, 2]
    assert body["trace"]["tokens"]["token_ids"] == [1, 2, 3]
    assert runtime.extract_calls == 1


def test_traces_endpoint_shape() -> None:
    _app, runtime = make_app()
    body = traces(
        ExtractRequest.model_validate({"model": "fake-model", "prompt": "hi", "extract": {"tokens": True}}),
        runtime=runtime,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
    )
    assert body["schema_version"] == "wllm.trace.v1"


def test_extraction_schema_is_generic_and_versioned() -> None:
    _app, runtime = make_app()
    body = extraction_schema(runtime=runtime, limits=ResourceLimits()).model_dump(mode="json")
    assert body["schema_version"] == "wllm.extraction.v1"
    assert body["request_schema"]["title"] == "ExtractRequest"
    assert "extract" in body["request_schema"]["properties"]
    logprobs_schema = body["request_schema"]["$defs"]["LogprobsExtraction"]
    assert logprobs_schema["properties"]["top_k"]["anyOf"][0]["minimum"] == 1
    assert logprobs_schema["properties"]["include_prompt"]["default"] is False
    payload = str(body).lower()
    assert "rauq" not in payload
    assert "eigenscore" not in payload
    assert "actmap" not in payload


def test_unsupported_extraction_error_envelope() -> None:
    _app, runtime = make_app()
    try:
        extract(
            ExtractRequest.model_validate(
                {
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
                }
            ),
            runtime=runtime,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
        )
    except UnsupportedExtractionError as exc:
        assert exc.status_code == 501
        assert exc.error_type == "unsupported_extraction"
    else:
        raise AssertionError("attention extraction should fail")


def test_served_model_name_reflected_in_models_and_chat() -> None:
    _app, runtime = make_app(served_model_name="alias-model")
    models = list_models(runtime=runtime)
    assert models["data"][0]["id"] == "alias-model"
    body = chat_completions(
        ChatCompletionRequest.model_validate(
            {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        ),
        runtime=runtime,
    )
    assert body["model"] == "alias-model"


def test_api_key_middleware_blocks_unauthorized_requests() -> None:
    from fastapi.testclient import TestClient

    runtime = FakeRuntime()
    app = create_app(runtime=runtime, api_key="secret")
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"

    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401

    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200


def test_api_key_middleware_allows_bearer_prefix_and_bare_token() -> None:
    from fastapi.testclient import TestClient

    runtime = FakeRuntime()
    app = create_app(runtime=runtime, api_key="secret")
    client = TestClient(app)
    assert client.get("/v1/models", headers={"Authorization": "secret"}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_app_without_api_key_allows_all_requests() -> None:
    from fastapi.testclient import TestClient

    runtime = FakeRuntime()
    app = create_app(runtime=runtime)
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
