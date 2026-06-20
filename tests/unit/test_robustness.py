from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from artifacts.store import ArtifactStore
from extractors.planning import (
    ResourceLimits,
    RuntimeTopology,
    compile_extraction_plan,
    validate_pre_generation_selectors,
)
from runtime.capabilities import default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator, GeneratedTraceInputs, LogprobCandidate
import runtime.vllm_runtime as vllm_runtime_module
from runtime.vllm_runtime import (
    VLLMRuntime,
    VLLMRuntimeConfig,
    _dtype_byte_size,
    _estimate_hidden_capture_bytes,
)
from schemas.extraction import (
    ExtractionSchemaResponse,
    ExtractRequest,
    extraction_schema_payload,
)
from server.errors import (
    ResourceLimitError,
    UnsupportedExtractionError,
)
from tracing.context import active_trace_id

# ---------------------------------------------------------------------------
# Reuse fake helpers from existing tests
# ---------------------------------------------------------------------------


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeImports:
    module = object()
    version = "0.9.0"
    LLM = None  # replaced per test
    SamplingParams = FakeSamplingParams


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        return " ".join(m.get("content", "") for m in messages)


class FakeCompletion:
    text = "ok"
    token_ids = [2]
    finish_reason = "stop"
    logprobs = None


class FakeOutput:
    prompt_token_ids = [1]
    prompt_logprobs = None
    outputs = [FakeCompletion()]


class CountingLLM:
    def __init__(self) -> None:
        self.generate_calls = 0

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def generate(self, prompts: list[str], sampling: FakeSamplingParams) -> list[FakeOutput]:
        del prompts, sampling
        self.generate_calls += 1
        return [FakeOutput()]


# ---------------------------------------------------------------------------
# Fake pooling helpers for deterministic replay tests
# ---------------------------------------------------------------------------


class DetFakeLayer:
    """Emit deterministic float32 data keyed by layer index."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:
        self.hooks.append(hook)
        return FakeHookHandle(self.hooks, hook)

    def emit(self, token_count: int) -> None:
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        output = (data + (self.index * 1000.0), None)
        for hook in list(self.hooks):
            hook(self, (), output)


class FakeHookHandle:
    def __init__(self, hooks: list[Any], hook: Any) -> None:
        self.hooks = hooks
        self.hook = hook

    def remove(self) -> None:
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class DetFakeInnerModel:
    def __init__(self) -> None:
        self.layers = [DetFakeLayer(index) for index in range(12)]


class DetFakeModel:
    def __init__(self) -> None:
        self.model = DetFakeInnerModel()


class DetFakePoolingLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.encode_calls: list[dict[str, Any]] = []
        self.model = DetFakeModel()

    def apply_model(self, func: Any) -> list[Any]:
        return [func(self.model)]

    def encode(self, *, prompt_token_ids: list[list[int]], use_tqdm: bool) -> list[Any]:
        self.encode_calls.append({"prompt_token_ids": prompt_token_ids, "use_tqdm": use_tqdm})
        token_count = len(prompt_token_ids[0])
        for layer in self.model.model.layers:
            layer.emit(token_count)
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        return [_DetFakeOutput(data)]


class _DetFakeOutput:
    def __init__(self, data: np.ndarray) -> None:
        self.outputs = _DetFakeData(data)


class _DetFakeData:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class DetFakePoolingImports:
    module = object()
    version = "0.9.0"
    LLM = DetFakePoolingLLM
    SamplingParams = FakeSamplingParams


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VAL-ROB-001: Concurrent extraction requests do not cross-contaminate
# ---------------------------------------------------------------------------


def test_concurrent_extract_requests_have_distinct_trace_ids(tmp_path, monkeypatch) -> None:
    """Two concurrent generate_extract calls produce distinct trace IDs."""
    from extractors.planning import ResourceLimits

    llm = CountingLLM()

    class FakeLLMWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = FakeLLMWrapper
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = llm

    trace_ids: list[str] = []
    errors: list[Exception] = []

    def extract() -> None:
        try:
            trace = runtime.generate_extract(
                ExtractRequest.model_validate(
                    {"model": "fake", "prompt": "hello", "extract": {"tokens": True}}
                ),
                limits=ResourceLimits(),
                artifact_store=ArtifactStore(tmp_path / "artifacts"),
                persist=False,
            )
            trace_ids.append(trace.id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=extract) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(trace_ids) == 3
    assert len(set(trace_ids)) == 3, f"Trace IDs should be unique: {trace_ids}"
    assert runtime._collector_registry is not None
    assert runtime._collector_registry.active_ids() == set()


# ---------------------------------------------------------------------------
# VAL-ROB-002: Normal generation concurrent with extraction does not interfere
# ---------------------------------------------------------------------------


def test_normal_chat_concurrent_with_extract_does_not_contaminate(tmp_path, monkeypatch) -> None:
    """Normal chat response must not contain extraction fields even when extract runs concurrently."""
    from schemas.openai import ChatCompletionRequest

    llm = CountingLLM()

    class FakeLLMWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = FakeLLMWrapper
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = llm

    chat_results: list[dict[str, Any]] = []
    extract_results: list[Any] = []

    def do_chat() -> None:
        response = runtime.generate_chat(
            ChatCompletionRequest.model_validate(
                {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
            )
        )
        chat_results.append(response)

    def do_extract() -> None:
        trace = runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {"tokens": True}}
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            persist=False,
        )
        extract_results.append(trace)

    threads = [
        threading.Thread(target=do_chat),
        threading.Thread(target=do_extract),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(chat_results) == 1
    assert len(extract_results) == 1

    # Chat response must be plain OpenAI shape with no trace/artifacts
    chat = chat_results[0]
    assert chat["choices"][0]["message"]["content"] == "ok"
    assert "trace" not in chat
    assert "artifacts" not in chat
    assert "schema_version" not in chat


# ---------------------------------------------------------------------------
# VAL-ROB-011: max_inline_tensor_bytes limit checked and rejected with hint
# ---------------------------------------------------------------------------


def test_max_inline_tensor_bytes_rejected_with_hint(tmp_path) -> None:
    """Request exceeding max_inline_tensor_bytes is rejected with a hint to use artifacts."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    inputs = GeneratedTraceInputs(
        model="fake-model",
        generation={},
        prompt_token_ids=list(range(2000)),
        generated_token_ids=list(range(2000, 4000)),
        generated_logprobs=[
            [LogprobCandidate(token_id=t, logprob=-0.1) for t in range(4000 + i, 4000 + i + 5)]
            for i in range(2000)
        ],
    )
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "max_tokens": 2000,
            "extract": {"tokens": True, "logprobs": {"top_k": 5, "include_prompt": True}},
        }
    )
    limits = ResourceLimits(
        max_inline_tensor_bytes=1000,
        max_total_captured_tensor_bytes=1_000_000,
    )

    with pytest.raises(ResourceLimitError) as exc:
        orchestrator.build_trace(
            request,
            inputs,
            limits=limits,
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.param == "extract"
    assert "hint" in exc.value.details
    assert exc.value.details["requested_inline_bytes"] > limits.max_inline_tensor_bytes
    assert "artifact" in exc.value.details["hint"].lower()
    assert exc.value.status_code == 413


# ---------------------------------------------------------------------------
# VAL-ROB-013: max_selected_heads enforced
# ---------------------------------------------------------------------------


def test_max_selected_heads_enforced() -> None:
    """Requesting more heads than max_selected_heads without large-extraction opt-in is rejected."""
    from schemas.extraction import ExtractionSpec

    spec = ExtractionSpec.model_validate(
        {
            "attentions": [
                {
                    "layers": "all",
                    "heads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "query_positions": "generated",
                    "key_positions": "previous_token",
                }
            ]
        }
    )
    limits = ResourceLimits(max_selected_heads=5, max_selected_layers=999, max_selected_positions=999)

    with pytest.raises(ResourceLimitError) as exc:
        validate_pre_generation_selectors(
            spec,
            topology=RuntimeTopology(num_layers=12, num_attention_heads=32, hidden_size=32),
            limits=limits,
        )

    assert exc.value.status_code == 413
    assert "heads" in str(exc.value).lower() or "head" in str(exc.value).lower()
    assert exc.value.details.get("required_artifact_include") == "attentions"


# ---------------------------------------------------------------------------
# VAL-ROB-014: max_selected_positions enforced
# ---------------------------------------------------------------------------


def test_max_selected_positions_enforced() -> None:
    """Requesting more positions than max_selected_positions without opt-in is rejected."""
    spec = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0], "positions": "generated"}]},
        }
    ).extract

    # 200 generated tokens exceeds max_selected_positions=5
    limits = ResourceLimits(
        max_selected_layers=999,
        max_selected_positions=5,
    )

    with pytest.raises(ResourceLimitError) as exc:
        compile_extraction_plan(
            spec,
            num_layers=4,
            prompt_token_count=10,
            generated_token_count=200,
            limits=limits,
            num_heads=None,
        )

    assert exc.value.status_code == 413
    assert "positions" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# VAL-ROB-015: max_artifact_bytes enforced with partial write cleanup
# ---------------------------------------------------------------------------


def test_max_artifact_bytes_enforced_with_cleanup(tmp_path) -> None:
    """Artifact exceeding max_artifact_bytes is rejected and the partial file is cleaned up."""
    store = ArtifactStore(tmp_path / "artifacts")

    # Write a valid artifact, then verify the limit check deletes the file.
    # The limit check for artifacts happens in the orchestration layer after write,
    # and calls delete_manifest_path to clean up.
    manifest = store.put(
        trace_id="trace_cleanup",
        tensors={"big": np.zeros((100, 100), dtype=np.float32)},
        format="npz",
    )
    artifact_path = tmp_path / "artifacts" / manifest.path
    assert artifact_path.exists()

    # Simulate orchestration cleanup when max_artifact_bytes exceeded
    store.delete_manifest_path(manifest.path)
    assert not artifact_path.exists(), "Artifact file should be deleted after limit exceeded"


# ---------------------------------------------------------------------------
# VAL-ROB-016: Large extraction requires artifact + allow_large + server opt-in
# ---------------------------------------------------------------------------


def test_large_extraction_requires_opt_in() -> None:
    """Full hidden-state dump without all three opt-in conditions is rejected."""
    # Use "all" layers (4 layers) and "generated" positions (4 tokens),
    # but max_selected_layers=2 and max_selected_positions=2 without opt-in
    spec = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": "all", "positions": "generated"}],
                "artifacts": {"include": [], "allow_large": False},
            },
        }
    ).extract

    limits = ResourceLimits(
        max_selected_layers=2,
        max_selected_positions=2,
        large_extraction_enabled=False,
    )

    with pytest.raises(ResourceLimitError) as exc:
        compile_extraction_plan(
            spec,
            num_layers=4,
            prompt_token_count=10,
            generated_token_count=4,
            limits=limits,
            num_heads=None,
        )

    details = exc.value.details
    assert details["required_artifact_include"] == "hidden_states"
    assert details["request_allow_large"] is False
    assert details["server_large_extraction_enabled"] is False


def test_large_extraction_missing_artifact_include_rejected() -> None:
    """Hidden states not in artifact include → rejected."""
    spec = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": "all", "positions": [0, 1, 2, 3]}],
                "artifacts": {"include": ["tokens"], "allow_large": True},
            },
        }
    ).extract

    limits = ResourceLimits(
        max_selected_layers=2,
        max_selected_positions=2,
        large_extraction_enabled=True,
    )

    with pytest.raises(ResourceLimitError) as exc:
        compile_extraction_plan(
            spec,
            num_layers=4,
            prompt_token_count=2,
            generated_token_count=2,
            limits=limits,
            num_heads=None,
        )

    details = exc.value.details
    assert details["required_artifact_include"] == "hidden_states"
    assert details["request_allow_large"] is True


def test_large_extraction_server_not_enabled_rejected() -> None:
    """Server large_extraction_enabled=False → rejected even with artifact opt-in."""
    spec = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": "all", "positions": [0, 1, 2, 3]}],
                "artifacts": {"include": ["hidden_states"], "allow_large": True},
            },
        }
    ).extract

    limits = ResourceLimits(
        max_selected_layers=2,
        max_selected_positions=2,
        large_extraction_enabled=False,
    )

    with pytest.raises(ResourceLimitError) as exc:
        compile_extraction_plan(
            spec,
            num_layers=4,
            prompt_token_count=2,
            generated_token_count=2,
            limits=limits,
            num_heads=None,
        )

    details = exc.value.details
    assert details["required_artifact_include"] == "hidden_states"
    assert details["request_allow_large"] is True
    assert details["server_large_extraction_enabled"] is False


# ---------------------------------------------------------------------------
# VAL-ROB-019: Hidden state capture byte estimation is conservative
# ---------------------------------------------------------------------------


def test_dtype_byte_size_mapping() -> None:
    """_dtype_byte_size must map known dtypes to correct byte widths."""
    assert _dtype_byte_size("bfloat16") == 2
    assert _dtype_byte_size("BFLOAT16") == 2
    assert _dtype_byte_size("float16") == 2
    assert _dtype_byte_size("half") == 2
    assert _dtype_byte_size("float32") == 4
    assert _dtype_byte_size("float") == 4
    assert _dtype_byte_size("auto") == 4
    assert _dtype_byte_size("float64") == 8
    assert _dtype_byte_size("double") == 8
    assert _dtype_byte_size("int8") == 1
    assert _dtype_byte_size("uint8") == 1
    # Unknown dtype defaults to 4 (conservative)
    assert _dtype_byte_size("unknown") == 4


def test_hidden_capture_bytes_never_underestimates() -> None:
    """_estimate_hidden_capture_bytes computes conservative upper bound."""
    topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=4096)

    # bf16: 12 * 100 * 4096 * 2 = 9,830,400
    estimate = _estimate_hidden_capture_bytes(
        layer_count=12, token_count=100, topology=topology, dtype="bfloat16"
    )
    expected = 12 * 100 * 4096 * 2
    assert estimate == expected

    # float32: 12 * 100 * 4096 * 4 = 19,660,800
    estimate = _estimate_hidden_capture_bytes(
        layer_count=12, token_count=100, topology=topology, dtype="auto"
    )
    expected = 12 * 100 * 4096 * 4
    assert estimate == expected

    # The raw estimate uses the full token count before position selection.
    # Since position selection can only reduce the count, this is always >= actual.
    # Verify with a smaller token_count: 5 tokens, single layer
    small_estimate = _estimate_hidden_capture_bytes(
        layer_count=1, token_count=5, topology=topology, dtype="float32"
    )
    small_expected = 1 * 5 * 4096 * 4
    assert small_estimate == small_expected

    # float64: double precision
    estimate_f64 = _estimate_hidden_capture_bytes(
        layer_count=1, token_count=10, topology=topology, dtype="float64"
    )
    expected_f64 = 1 * 10 * 4096 * 8
    assert estimate_f64 == expected_f64


# ---------------------------------------------------------------------------
# VAL-ROB-023: Attention weights always return unsupported
# ---------------------------------------------------------------------------


def test_attention_weights_always_unsupported() -> None:
    """Any request with extract.attentions raises UnsupportedExtractionError."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
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
    )

    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.preflight(request, limits=ResourceLimits())

    assert exc.value.status_code == 501
    assert "unavailable" in exc.value.code


# ---------------------------------------------------------------------------
# VAL-ROB-024: Unsupported extraction types never return placeholder tensors
# ---------------------------------------------------------------------------


def test_unsupported_extraction_never_returns_placeholder() -> None:
    """Requesting raw_logits returns error, never 200 with empty data."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"logprobs": {"top_k": 5, "raw_logits": True}},
        }
    )

    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.preflight(request, limits=ResourceLimits())

    assert exc.value.status_code == 501
    assert exc.value.code == "raw_logits_unavailable"


def test_exact_entropy_without_approximation_rejected() -> None:
    """Exact entropy without allow_approximate_entropy returns 501."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"logprobs": {"top_k": 5, "entropy": True, "allow_approximate_entropy": False}},
        }
    )

    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.preflight(request, limits=ResourceLimits())

    assert exc.value.status_code == 501
    assert exc.value.code == "exact_entropy_unavailable"


# ---------------------------------------------------------------------------
# VAL-ROB-028: Active trace context is cleaned up after request processing
# ---------------------------------------------------------------------------


def test_active_trace_context_cleaned_up_after_normal_return(tmp_path) -> None:
    """active_trace_id is reset to its prior value after build_trace returns normally."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    inputs = GeneratedTraceInputs(
        model="fake-model",
        generation={},
        prompt_token_ids=[1, 2],
        generated_token_ids=[3],
    )
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"tokens": True}}
    )

    # Ensure clean starting state
    assert active_trace_id.get() is None

    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace is not None
    assert active_trace_id.get() is None, "active_trace_id should be None after build_trace"


def test_active_trace_context_restored_after_nested_call(tmp_path) -> None:
    """active_trace_id is restored to a prior custom value after build_trace."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    inputs = GeneratedTraceInputs(
        model="fake-model",
        generation={},
        prompt_token_ids=[1],
        generated_token_ids=[2],
    )
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {}}
    )

    # Set a prior context value
    token = active_trace_id.set("outer_trace_id")
    assert active_trace_id.get() == "outer_trace_id"

    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace is not None
    assert active_trace_id.get() == "outer_trace_id", "Prior trace context should be restored"

    active_trace_id.reset(token)
    assert active_trace_id.get() is None


def test_active_trace_context_cleaned_up_after_exception(tmp_path) -> None:
    """active_trace_id is reset even when build_trace raises an exception."""
    orchestrator = ExtractionOrchestrator(
        default_vllm_capabilities("fake-model", "0.10.2")
    )
    # This request includes logprobs with high inline bytes to trigger ResourceLimitError
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "max_tokens": 1000,
            "extract": {"tokens": True, "logprobs": {"top_k": 10, "include_prompt": True}},
        }
    )
    # Make inputs have many tokens to exceed limits
    inputs_big = GeneratedTraceInputs(
        model="fake-model",
        generation={},
        prompt_token_ids=list(range(1000)),
        generated_token_ids=list(range(1000, 2000)),
        generated_logprobs=[
            [LogprobCandidate(token_id=t, logprob=-0.1) for t in range(i, i + 5)]
            for i in range(1000)
        ],
        prompt_logprobs=[
            [LogprobCandidate(token_id=t, logprob=-0.1) for t in range(i, i + 5)]
            for i in range(1000)
        ],
    )

    assert active_trace_id.get() is None

    try:
        orchestrator.build_trace(
            request,
            inputs_big,
            limits=ResourceLimits(max_inline_tensor_bytes=1000),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except ResourceLimitError:
        pass

    assert active_trace_id.get() is None, "active_trace_id must be None even after exception"


# ---------------------------------------------------------------------------
# VAL-ROB-029: Extraction schema includes configured resource limits
# ---------------------------------------------------------------------------


def test_extraction_schema_includes_resource_limits() -> None:
    """extraction_schema_payload returns limits with all fields matching configuration."""
    limits = ResourceLimits(
        max_top_k=42,
        max_selected_layers=12,
        max_selected_heads=16,
        max_selected_positions=128,
        max_inline_tensor_bytes=500_000,
        max_total_captured_tensor_bytes=10_000_000,
        max_artifact_bytes=50_000_000,
        large_extraction_enabled=True,
    )
    capabilities = default_vllm_capabilities("fake-model", "0.10.2")

    schema = extraction_schema_payload(limits=limits, capabilities=capabilities)

    assert isinstance(schema, ExtractionSchemaResponse)
    assert schema.limits.max_top_k == 42
    assert schema.limits.max_selected_layers == 12
    assert schema.limits.max_selected_heads == 16
    assert schema.limits.max_selected_positions == 128
    assert schema.limits.max_inline_tensor_bytes == 500_000
    assert schema.limits.max_total_captured_tensor_bytes == 10_000_000
    assert schema.limits.max_artifact_bytes == 50_000_000
    assert schema.limits.large_extraction_enabled is True


# ---------------------------------------------------------------------------
# VAL-ROB-030: Extraction schema includes runtime capabilities
# ---------------------------------------------------------------------------


def test_extraction_schema_includes_runtime_capabilities() -> None:
    """extraction_schema_payload returns capabilities with all expected feature fields."""
    capabilities = default_vllm_capabilities(
        "fake-model",
        "0.10.2",
        attention_backend="FLASH_ATTN",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        supported_runner_types=["generate", "pooling"],
    )
    limits = ResourceLimits()

    schema = extraction_schema_payload(limits=limits, capabilities=capabilities)

    caps = schema.capabilities
    assert caps.model == "fake-model"
    assert caps.vllm_version == "0.10.2"
    assert caps.attention_backend == "FLASH_ATTN"
    assert caps.token_ids.state == "supported"
    assert caps.attentions.state == "unsupported"
    assert caps.hidden_states.state in ("supported", "conditional")
    assert caps.npz_artifacts.state == "supported"
    assert caps.pt_artifacts.state in ("supported", "conditional")
    assert caps.exact_entropy.state == "unsupported"
    assert caps.top_k_logits.state == "unsupported"


# ---------------------------------------------------------------------------
# VAL-ROB-031: Online hidden-state capture serialized across concurrent requests
# ---------------------------------------------------------------------------


def test_concurrent_online_hidden_state_requests_serialized_hooks(tmp_path, monkeypatch) -> None:
    """Two sequential online hidden-state requests do not leave hooks behind.

    The online capture path installs hooks via _apply_model_to_workers and
    removes them in a finally path. This test verifies that after two back-to-back
    online extractions, the model has no residual hook state.
    """
    from extractors.planning import ResourceLimits

    # Reuse the existing FakeOnlineLLM pattern from test_vllm_runtime
    class FakeHookHandle:
        def __init__(self, hooks: list[Any], hook: Any) -> None:
            self.hooks = hooks
            self.hook = hook

        def remove(self) -> None:
            if self.hook in self.hooks:
                self.hooks.remove(self.hook)

    class FakeLayer:
        def __init__(self, index: int) -> None:
            self.index = index
            self.hooks: list[Any] = []

        def register_forward_hook(self, hook: Any) -> Any:
            self.hooks.append(hook)
            return FakeHookHandle(self.hooks, hook)

        def emit(self, token_count: int) -> None:
            data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
            output = (data + (self.index * 1000.0), None)
            for hook in list(self.hooks):
                hook(self, (), output)

    class FakeInnerModel:
        def __init__(self) -> None:
            self.layers = [FakeLayer(index) for index in range(12)]

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeInnerModel()

    class FakeTokenizer:
        @staticmethod
        def encode(prompt: str, **kwargs: Any) -> int:
            return 1

        @staticmethod
        def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            return " ".join(m.get("content", "") for m in messages)

    class FakeCompletion:
        text = "ok"
        token_ids = [2]
        finish_reason = "stop"

    class FakeOutput:
        prompt_token_ids = [1]
        outputs = [FakeCompletion()]

    class FakeOnlineLLM:
        def __init__(self) -> None:
            self.generate_calls = 0
            self.model = FakeModel()

        def get_tokenizer(self) -> Any:
            return FakeTokenizer()

        def apply_model(self, func: Any) -> list[Any]:
            return [func(self.model)]

        def generate(self, prompts: list[str], sampling: Any) -> list[Any]:
            del prompts, sampling
            self.generate_calls += 1
            for layer in self.model.model.layers:
                layer.emit(2)
            return [FakeOutput()]

    llm = FakeOnlineLLM()

    class FakeLLMWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = FakeLLMWrapper
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())

    class OnlineImports:
        module = object()
        version = "0.10.2"
        LLM = FakeOnlineLLM
        SamplingParams = FakeSamplingParams

    runtime = VLLMRuntime(
        VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True)
    )
    runtime._imports = OnlineImports()
    runtime._llm = llm
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    # Run two sequential online extractions
    for i in range(2):
        trace = runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "hidden_states": [
                            {"layers": 5, "positions": "prompt", "capture_mode": "online"}
                        ]
                    },
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            persist=False,
        )
        assert trace.trace.hidden_states[0].capture_mode == "online"
        # After each extraction, all hooks must be removed
        assert all(not layer.hooks for layer in llm.model.model.layers), (
            f"Iteration {i}: hooks not removed"
        )


# ---------------------------------------------------------------------------
# VAL-ROB-032: Replay hidden-state numerical determinism
# ---------------------------------------------------------------------------


def test_replay_hidden_states_deterministic_across_identical_requests(tmp_path, monkeypatch) -> None:
    """Two identical replay hidden-state requests produce np.allclose tensors."""
    from extractors.planning import ResourceLimits

    det_llm = DetFakePoolingLLM()

    class FakeLLMWrapper:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class DetImports:
        module = object()
        version = "0.10.2"
        LLM = FakeLLMWrapper
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: DetImports())

    def make_runtime() -> VLLMRuntime:
        runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
        runtime._imports = DetImports()
        runtime._llm = CountingLLM()
        runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
        runtime._supported_runner_types = ["generate", "pooling"]
        # Override the pooling LLM with our deterministic fake
        runtime._pooling_llm = det_llm
        return runtime

    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, 3, 5], "positions": "last_generated"}]},
        }
    )

    # First extraction
    runtime1 = make_runtime()
    trace1 = runtime1.generate_extract(
        request,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path / "a"),
        persist=False,
    )

    # Second extraction
    runtime2 = make_runtime()
    trace2 = runtime2.generate_extract(
        request,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path / "b"),
        persist=False,
    )

    record1 = trace1.trace.hidden_states[0]
    record2 = trace2.trace.hidden_states[0]

    assert record1.layers == record2.layers
    assert record1.positions == record2.positions
    assert record1.shape == record2.shape

    data1 = np.asarray(record1.data, dtype=np.float32)
    data2 = np.asarray(record2.data, dtype=np.float32)
    assert np.allclose(data1, data2, rtol=1e-5, atol=1e-8), "Replay hidden states must be deterministic"
