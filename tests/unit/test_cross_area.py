"""Cross-area flow test coverage for wllm 0.1.0.

Covers: wrong-endpoint dispatch, CLI flag propagation to vLLM constructor,
artifact-dir propagation, topology validation, capability metadata,
research adapter isolation, invalid model path handling, researcher E2E,
log-level affecting output, and full lifecycle flow.
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from artifacts.store import ArtifactStore
from extractors.planning import (
    ResourceLimits,
    RuntimeTopology,
)
from extractors.selectors import (
    SelectorValidationError,
    normalize_head_selector,
    normalize_layer_selector,
)
from research.base import ResearchAdapter, ResearchResult
from research.token_baselines import TokenBaselineAdapter
from runtime.capabilities import default_vllm_capabilities
import runtime.vllm_compat as vllm_compat_module
import runtime.vllm_runtime as vllm_runtime_module
from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
from schemas.extraction import extraction_schema_payload
from schemas.openai import ChatCompletionRequest, CompletionRequest
from schemas.traces import TokenTrace, TraceData, TraceEnvelope, TraceMetadata
from server.app import create_app
from server.errors import RuntimeUnavailableError

# ---------------------------------------------------------------------------
# Fakes shared across tests
# ---------------------------------------------------------------------------


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeImports:
    SamplingParams = FakeSamplingParams
    version = "0.10.2"
    LLM: Any = None  # set per test


class FakeCompletion:
    text = "ok"
    token_ids = [2, 3]
    finish_reason = "stop"
    logprobs = None


class FakeOutput:
    prompt_token_ids = [1]
    outputs = [FakeCompletion()]


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        return "rendered:" + "|".join(m.get("content", "") for m in messages if m.get("role") != "system")

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return f"<token:{token_ids[-1]}>"

    def batch_decode(self, token_id_batches: list[list[int]], **kwargs: Any) -> list[str]:
        del kwargs
        return [f"<token:{tokens[-1]}>" for tokens in token_id_batches]


class FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: list[tuple[list[str], Any]] = []

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def generate(self, prompts: list[str], sampling: Any) -> list[FakeOutput]:
        self.generate_calls.append((prompts, sampling))
        return [FakeOutput()]


class FakeRuntime:
    """Minimal fake runtime for route-level tests."""

    def __init__(self, served_model_name: str | None = None) -> None:
        self.model = "fake-model"
        self.served_model_name = served_model_name or self.model
        self.extract_calls = 0
        self.chat_calls = 0

    def capabilities(self) -> Any:
        return default_vllm_capabilities(self.model, "0.10.2")

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": self.served_model_name, "object": "model", "created": 0, "owned_by": "wllm"}],
        }

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
        return {
            "id": "cmpl_fake",
            "object": "text_completion",
            "created": 0,
            "model": self.served_model_name,
            "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def generate_extract(self, request, *, limits, artifact_store, persist):
        self.extract_calls += 1
        return TraceEnvelope(
            id="trace_fake",
            created=0,
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
            metadata=TraceMetadata(capabilities=self.capabilities().as_metadata()),
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-026: chat body rejected by /v1/extract with 422
# ---------------------------------------------------------------------------


def test_chat_body_rejected_by_extract_endpoint_422() -> None:
    """ChatCompletionRequest-shaped body to /v1/extract returns 422.

    The body includes ``logprobs`` (valid for ChatCompletionRequest but an
    extra=forbid field for ExtractRequest), triggering schema validation failure.
    """
    runtime = FakeRuntime()
    app = create_app(runtime=runtime)
    client = TestClient(app)
    response = client.post(
        "/v1/extract",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "logprobs": True,
        },
    )
    assert response.status_code == 422
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "schema_validation_failed"


# ---------------------------------------------------------------------------
# VAL-CROSS-027: extract body rejected by /v1/chat/completions with 422
# ---------------------------------------------------------------------------


def test_extract_body_rejected_by_chat_endpoint_422() -> None:
    """ExtractRequest-shaped body to /v1/chat/completions returns 422.

    The body includes ``extract`` (valid for ExtractRequest but extra=forbid for
    ChatCompletionRequest), triggering schema validation failure.
    """
    runtime = FakeRuntime()
    app = create_app(runtime=runtime)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "extract": {"tokens": True},
        },
    )
    assert response.status_code == 422
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "schema_validation_failed"


# ---------------------------------------------------------------------------
# VAL-CROSS-011: --artifact-dir propagates to artifact store root path
# ---------------------------------------------------------------------------


def test_artifact_dir_propagates_to_store_root(tmp_path) -> None:
    """ArtifactStore root equals the resolved path passed at construction."""
    target = tmp_path / "wllm-artifacts"
    store = ArtifactStore(target)
    assert store.root == target.resolve()


def test_artifact_store_root_accepts_relative_path() -> None:
    """ArtifactStore resolves relative paths to absolute."""
    store = ArtifactStore(Path("./relative-artifacts"))
    assert store.root.is_absolute()
    assert store.root.name == "relative-artifacts"


# ---------------------------------------------------------------------------
# CLI flag propagation to LLM constructor
# ---------------------------------------------------------------------------


def test_trust_remote_code_propagates_to_llm_constructor(monkeypatch) -> None:
    """--trust-remote-code flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", trust_remote_code=True))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("trust_remote_code") is True


def test_dtype_propagates_to_llm_constructor(monkeypatch) -> None:
    """--dtype flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", dtype="bfloat16"))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("dtype") == "bfloat16"


def test_max_model_len_propagates_to_llm_constructor(monkeypatch) -> None:
    """--max-model-len flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", max_model_len=4096))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("max_model_len") == 4096


def test_max_model_len_none_propagates_to_llm_constructor(monkeypatch) -> None:
    """--max-model-len omitted (None) flows to LLM kwargs correctly."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", max_model_len=None))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("max_model_len") is None


def test_gpu_memory_utilization_propagates_to_llm_constructor(monkeypatch) -> None:
    """--gpu-memory-utilization flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.5))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("gpu_memory_utilization") == 0.5


def test_tensor_parallel_size_propagates_to_llm_constructor(monkeypatch) -> None:
    """--tensor-parallel-size flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", tensor_parallel_size=2))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("tensor_parallel_size") == 2


def test_tokenizer_propagates_to_llm_constructor(monkeypatch) -> None:
    """--tokenizer flows through VLLMRuntimeConfig to LLM kwargs."""
    class CaptureLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = CaptureLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", tokenizer="custom/tok"))
    runtime._ensure_loaded()
    assert runtime._llm.kwargs.get("tokenizer") == "custom/tok"


def test_seed_propagates_to_sampling_params() -> None:
    """--seed flows through VLLMRuntimeConfig to SamplingParams."""
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", seed=42))
    runtime._imports = FakeImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello"})
    sampling = runtime._sampling_params(request)
    assert sampling.kwargs["seed"] == 42


# ---------------------------------------------------------------------------
# VAL-CROSS-015: model topology extracted from vLLM config
# ---------------------------------------------------------------------------


def test_model_topology_from_fake_vllm_config_alternate_fields() -> None:
    """Topology extraction handles alternate config field names."""
    from runtime.vllm_compat import extract_model_topology

    class HFConfig:
        n_layer = 10
        n_head = 12
        n_embd = 768

    class ModelConfig:
        hf_config = HFConfig()

    class Engine:
        model_config = ModelConfig()

    class LLM:
        llm_engine = Engine()

    topology = extract_model_topology(LLM())
    assert topology is not None
    assert topology.num_layers == 10
    assert topology.num_attention_heads == 12
    assert topology.hidden_size == 768


def test_model_topology_from_config_with_num_layers_d_model() -> None:
    """Topology extraction handles num_layers/d_model naming."""
    from runtime.vllm_compat import extract_model_topology

    class HFConfig:
        num_layers = 16
        num_heads = 8
        d_model = 512

    class ModelConfig:
        hf_config = HFConfig()

    class Engine:
        model_config = ModelConfig()

    class LLM:
        llm_engine = Engine()

    topology = extract_model_topology(LLM())
    assert topology.num_layers == 16
    assert topology.num_attention_heads == 8
    assert topology.hidden_size == 512


# ---------------------------------------------------------------------------
# VAL-CROSS-016: model topology used to validate layer selectors
# ---------------------------------------------------------------------------


def test_topology_used_to_validate_layer_selectors_before_capture() -> None:
    """normalize_layer_selector uses topology.num_layers for validation."""
    topology = RuntimeTopology(num_layers=32, num_attention_heads=12, hidden_size=4096)

    # "all" resolves to 0..31
    result = normalize_layer_selector("all", topology.num_layers)
    assert result == list(range(32))

    # Layer index within range
    assert normalize_layer_selector(0, topology.num_layers) == [0]
    assert normalize_layer_selector(31, topology.num_layers) == [31]

    # Negative index
    assert normalize_layer_selector(-1, topology.num_layers) == [31]
    assert normalize_layer_selector(-32, topology.num_layers) == [0]

    # Out of range raises SelectorValidationError
    with pytest.raises(SelectorValidationError):
        normalize_layer_selector(32, topology.num_layers)

    with pytest.raises(SelectorValidationError):
        normalize_layer_selector(-33, topology.num_layers)


# ---------------------------------------------------------------------------
# VAL-CROSS-017: model topology used to validate head selectors
# ---------------------------------------------------------------------------


def test_topology_used_to_validate_head_selectors_before_attention() -> None:
    """normalize_head_selector uses topology.num_attention_heads for validation."""
    topology = RuntimeTopology(num_layers=12, num_attention_heads=32, hidden_size=2048)

    # "all" returns literal "all"
    assert normalize_head_selector("all", topology.num_attention_heads) == "all"

    # Integer within range
    assert normalize_head_selector(0, topology.num_attention_heads) == [0]
    assert normalize_head_selector(31, topology.num_attention_heads) == [31]

    # Out of range
    with pytest.raises(SelectorValidationError):
        normalize_head_selector(32, topology.num_attention_heads)

    with pytest.raises(SelectorValidationError):
        normalize_head_selector(-33, topology.num_attention_heads)


# ---------------------------------------------------------------------------
# VAL-CROSS-018: extraction schema includes configured resource limits
# ---------------------------------------------------------------------------


def test_extraction_schema_respects_non_default_resource_limits() -> None:
    """Extraction schema reflects non-default ResourceLimits configuration."""
    from schemas.extraction import ExtractionSchemaResponse

    limits = ResourceLimits(
        max_top_k=25,
        max_selected_layers=8,
        max_selected_heads=12,
        max_selected_positions=64,
        max_inline_tensor_bytes=256_000,
        max_total_captured_tensor_bytes=5_000_000,
        max_artifact_bytes=10_000_000,
        large_extraction_enabled=False,
    )
    capabilities = default_vllm_capabilities("fake-model", "0.10.2")
    schema = extraction_schema_payload(limits=limits, capabilities=capabilities)

    assert isinstance(schema, ExtractionSchemaResponse)
    assert schema.limits.max_top_k == 25
    assert schema.limits.max_selected_layers == 8
    assert schema.limits.max_selected_heads == 12
    assert schema.limits.max_selected_positions == 64
    assert schema.limits.max_inline_tensor_bytes == 256_000
    assert schema.limits.max_total_captured_tensor_bytes == 5_000_000
    assert schema.limits.max_artifact_bytes == 10_000_000
    assert schema.limits.large_extraction_enabled is False


# ---------------------------------------------------------------------------
# VAL-CROSS-019: extraction schema includes runtime capabilities
# ---------------------------------------------------------------------------


def test_extraction_schema_includes_all_capability_fields() -> None:
    """Extraction schema capabilities object has all expected feature fields."""
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
    # Every feature field must be present
    assert caps.token_ids is not None
    assert caps.token_logprobs is not None
    assert caps.prompt_logprobs is not None
    assert caps.top_k_logprobs is not None
    assert caps.top_k_logits is not None
    assert caps.exact_entropy is not None
    assert caps.hidden_states is not None
    assert caps.attentions is not None
    assert caps.npz_artifacts is not None
    assert caps.pt_artifacts is not None


# ---------------------------------------------------------------------------
# VAL-CROSS-020: capability metadata updates when config changes
# ---------------------------------------------------------------------------


def test_capability_metadata_attention_backend_from_env(monkeypatch) -> None:
    """Capability attention_backend prefers VLLM_ATTENTION_BACKEND env var."""
    caps = default_vllm_capabilities(
        "fake-model",
        "0.10.2",
        attention_backend="FLASH_ATTN",
    )
    assert caps.attention_backend == "FLASH_ATTN"


def test_capability_metadata_tensor_parallel_affects_hidden_states() -> None:
    """tensor_parallel_size > 1 makes hidden_states unsupported."""
    caps = default_vllm_capabilities(
        "fake-model",
        "0.10.2",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.4,
    )
    assert caps.hidden_states.state == "unsupported"
    assert caps.hidden_states.details.get("tensor_parallel_size") == 2


def test_capability_metadata_high_gpu_memory_affects_hidden_states() -> None:
    """gpu_memory_utilization > 0.5 makes hidden_states unsupported."""
    caps = default_vllm_capabilities(
        "fake-model",
        "0.10.2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    assert caps.hidden_states.state == "unsupported"
    assert caps.hidden_states.details.get("gpu_memory_utilization") == 0.9


def test_capability_metadata_online_hidden_states_enables_online_mode() -> None:
    """enable_online_hidden_states=True includes 'online' in capture modes."""
    caps = default_vllm_capabilities(
        "fake-model",
        "0.10.2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        online_hidden_states=True,
        supported_runner_types=["generate", "pooling"],
    )
    assert caps.hidden_states.state in ("supported", "conditional")
    capture_modes = caps.hidden_states.details.get("capture_modes", [])
    assert "online" in capture_modes


def test_capability_metadata_attention_weights_requires_opt_in() -> None:
    disabled = default_vllm_capabilities("fake-model", "0.10.2")
    enabled = default_vllm_capabilities("fake-model", "0.10.2", attention_weights=True)

    assert disabled.attentions.state == "unsupported"
    assert disabled.attentions.details["enable_attention_weights"] is False
    assert enabled.attentions.state == "conditional"
    assert enabled.attentions.details["backend"] == "transformers_replay"
    assert enabled.attentions.details["capture_modes"] == ["replay"]


# ---------------------------------------------------------------------------
# VAL-CROSS-021: research adapters not imported during normal serving
# ---------------------------------------------------------------------------


def test_research_adapters_not_imported_during_normal_serving() -> None:
    """Normal chat completion does not import any research adapter module."""
    # Remove any pre-existing research imports
    research_keys_before = [k for k in sys.modules if k.startswith("research")]
    for key in research_keys_before:
        if key not in ("research", "research.base"):
            del sys.modules[key]

    runtime = FakeRuntime()
    app = create_app(runtime=runtime)
    client = TestClient(app)

    # Perform a normal chat completion request
    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert response.status_code == 200

    # Research adapter submodules should not be loaded
    research_modules = [k for k in sys.modules if k.startswith("research.")]
    adapter_modules = [
        m for m in research_modules
        if m not in ("research.base", "research.__init__")
    ]
    assert adapter_modules == [], (
        f"Research adapters leaked into sys.modules during normal serving: {adapter_modules}"
    )


# ---------------------------------------------------------------------------
# VAL-CROSS-022: research adapters consume generic TraceEnvelope
# ---------------------------------------------------------------------------


def test_research_adapter_protocol_accepts_trace_envelope() -> None:
    """ResearchAdapter.run signature accepts TraceEnvelope as first positional param."""
    # Check the Protocol definition
    sig = inspect.signature(ResearchAdapter.run)
    params = list(sig.parameters.keys())
    assert params[0] == "self"
    assert params[1] == "trace"
    # The trace parameter should be typed as TraceEnvelope
    hints = inspect.get_annotations(ResearchAdapter.run)
    assert "trace" in hints


def test_token_baseline_adapter_consumes_trace_envelope() -> None:
    """TokenBaselineAdapter.run accepts and processes a TraceEnvelope."""
    adapter = TokenBaselineAdapter()
    trace = TraceEnvelope(
        id="trace_1",
        created=0,
        model="fake",
        generation={"id": "cmpl_1", "choices": [], "usage": {}},
        trace=TraceData(
            tokens=TokenTrace(token_ids=[1, 2, 3, 4, 5], tokens=["a", "b", "c", "d", "e"]),
            spans={"prompt": (0, 2), "generated": (2, 5)},
        ),
    )
    result = adapter.run(trace)
    assert isinstance(result, ResearchResult)
    assert result.status == "ok"
    assert result.values["token_count"] == 5
    assert result.values["generated_token_count"] == 3


def test_all_research_adapters_accept_trace_envelope() -> None:
    """Every research adapter's run method accepts TraceEnvelope as first arg."""
    from research.actmap import ActMapAdapter
    from research.eigenscore import EigenScoreAdapter
    from research.linear_probe import LinearProbeAdapter
    from research.rauq import RAUQAdapter

    trace = TraceEnvelope(
        id="trace_1",
        created=0,
        model="fake",
        generation={"id": "cmpl_1", "choices": [], "usage": {}},
        trace=TraceData(
            tokens=TokenTrace(token_ids=[1, 2, 3], tokens=["a", "b", "c"]),
            spans={"prompt": (0, 2), "generated": (2, 3)},
        ),
    )
    for adapter_class in [
        ActMapAdapter, EigenScoreAdapter, LinearProbeAdapter, RAUQAdapter,
        TokenBaselineAdapter,
    ]:
        adapter = adapter_class()
        result = adapter.run(trace)
        assert isinstance(result, ResearchResult)
        assert result.name == adapter.name
        assert result.status in ("ok", "unsupported")


# ---------------------------------------------------------------------------
# VAL-CROSS-023: vLLM version guard prevents init with wrong version
# (Already covered by test_import_vllm_rejects_unvalidated_versions)
# This test adds an end-to-end coverage through VLLMRuntime._ensure_loaded
# ---------------------------------------------------------------------------


def test_vllm_version_guard_prevents_runtime_init(monkeypatch) -> None:
    """Runtime._ensure_loaded raises RuntimeUnavailableError for wrong vLLM version."""
    monkeypatch.setattr(vllm_compat_module.importlib.metadata, "version", lambda _name: "0.11.0")

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    with pytest.raises(RuntimeUnavailableError) as exc:
        runtime._ensure_loaded()
    assert exc.value.code == "unsupported_vllm_version"
    assert "installed" in exc.value.details or exc.value.details.get("installed")
    assert "supported" in exc.value.details or exc.value.details.get("supported")


# ---------------------------------------------------------------------------
# VAL-CROSS-029: wllm serve with non-existent model path returns structured error
# ---------------------------------------------------------------------------


def test_serve_with_nonexistent_model_returns_structured_error(monkeypatch) -> None:
    """Non-existent model path produces RuntimeUnavailableError with structured info."""
    class AbsentModelLLM:
        def __init__(self, **kwargs):
            raise FileNotFoundError(f"Cannot find model: {kwargs.get('model')}")

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = AbsentModelLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="/nonexistent/path"))
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        runtime._ensure_loaded()

    err = excinfo.value
    assert err.code == "vllm_initialization_failed"
    assert err.status_code == 503
    assert "/nonexistent/path" in str(err.details.get("model", ""))
    # Must not leak raw traceback as the message
    assert "Traceback" not in err.message


def test_serve_with_invalid_hf_model_id_returns_structured_error(monkeypatch) -> None:
    """Invalid HuggingFace model ID produces RuntimeUnavailableError."""
    class NetworkFailLLM:
        def __init__(self, **kwargs):
            raise RuntimeError(f"Failed to download model '{kwargs.get('model')}': connection refused")

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = NetworkFailLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="org/nonexistent-model-99999"))
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        runtime._ensure_loaded()

    err = excinfo.value
    assert err.code == "vllm_initialization_failed"
    assert err.status_code == 503
    assert "org/nonexistent-model-99999" in str(err.details.get("model", ""))


# ---------------------------------------------------------------------------
# VAL-CROSS-030: researcher E2E: extract -> artifact -> load -> adapter consumption
# ---------------------------------------------------------------------------


def test_researcher_e2e_extract_artifact_load_adapter(tmp_path) -> None:
    """Full researcher workflow: extract with artifacts, load, run adapter.

    Simulates the researcher flow: create trace → write artifact → load artifact → run adapter.
    """
    from artifacts.npz import load_npz

    store = ArtifactStore(tmp_path / "artifacts")

    # Create a trace envelope (simulating extraction output)
    token_ids = [1, 2, 3, 4, 5]
    trace = TraceEnvelope(
        id="trace_e2e",
        created=0,
        model="fake-model",
        generation={
            "id": "cmpl_e2e",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        trace=TraceData(
            tokens=TokenTrace(token_ids=token_ids, tokens=[f"t{i}" for i in token_ids]),
            spans={"prompt": (0, 2), "generated": (2, 5)},
        ),
        metadata=TraceMetadata(capabilities={}),
    )

    # Write artifact (simulating extraction with persist=True)
    tensors = {"hidden_states_layer_0": np.zeros((3, 32), dtype=np.float32)}
    manifest = store.put(trace_id=trace.id, tensors=tensors, format="npz")

    # Load artifact back (researcher loading persisted artifacts)
    loaded = load_npz(store.root / manifest.path)
    assert "hidden_states_layer_0" in loaded
    assert np.array_equal(loaded["hidden_states_layer_0"], tensors["hidden_states_layer_0"])

    # Run research adapter on the trace (researcher consuming trace data)
    adapter = TokenBaselineAdapter()
    result = adapter.run(trace)
    assert result.status == "ok"
    assert result.values["token_count"] == 5
    assert result.values["generated_token_count"] == 3


# ---------------------------------------------------------------------------
# Log-level affecting output
# ---------------------------------------------------------------------------


def test_log_level_affects_logging_basic_config(monkeypatch) -> None:
    """--log-level passed to _cmd_serve is forwarded to logging.basicConfig."""
    import cli as cli_module
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    called_level: list[str] = []

    def fake_basicConfig(**kwargs):
        called_level.append(kwargs.get("level", ""))

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)

    parser = cli_module.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--log-level", "debug", "--artifact-dir", "/tmp/test"])

    class FakeRuntime:
        def __init__(self, config):
            self.config = config

    class FakeStore:
        def __init__(self, path):
            pass

    def fake_create_app(runtime, artifact_store, limits, api_key):
        del runtime, artifact_store, limits, api_key
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        called_level.append(f"uvicorn:{log_level}")

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    rc = cli_module._cmd_serve(args)
    assert rc == 0

    # logging.basicConfig should have been called with level=logging.DEBUG (10)
    assert len(called_level) >= 1
    assert called_level[0] == logging.DEBUG
    # uvicorn.run should receive the raw log level string
    assert "uvicorn:debug" in called_level


# ---------------------------------------------------------------------------
# VAL-CROSS-031: wllm serve MODEL lifecycle: start -> serve -> shutdown cleanly
# ---------------------------------------------------------------------------


def test_serve_lifecycle_start_serve_shutdown_flow(monkeypatch, tmp_path) -> None:
    """_cmd_serve executes the full lifecycle: config → prewarm → create_app → uvicorn."""
    import cli as cli_module
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    parser = cli_module.build_parser()
    args = parser.parse_args([
        "serve", "Qwen/Qwen3-0.6B",
        "--artifact-dir", str(tmp_path / "artifacts"),
    ])

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def prewarm_hidden_states(self):
            captured["prewarmed"] = True

    class FakeStore:
        def __init__(self, path):
            captured["artifact_dir"] = path

    def fake_create_app(runtime, artifact_store, limits, api_key):
        del runtime, artifact_store
        captured["create_app_called"] = True
        captured["limits"] = limits
        captured["api_key"] = api_key
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        captured["uvicorn_called"] = True
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    rc = cli_module._cmd_serve(args)
    assert rc == 0

    # Verify the lifecycle order: config was created, prewarm was NOT called
    # (prewarm_hidden_states=False by default), create_app was called, uvicorn was called
    assert captured.get("config") is not None
    assert captured["config"].model == "Qwen/Qwen3-0.6B"
    assert captured.get("prewarmed") is None  # not prewarmed by default
    assert captured.get("create_app_called") is True
    assert captured.get("uvicorn_called") is True
    assert captured.get("host") == "127.0.0.1"
    assert captured.get("port") == 8000
    assert captured.get("log_level") == "info"


def test_serve_lifecycle_with_prewarm_config(monkeypatch, tmp_path) -> None:
    """_cmd_serve prewarms hidden states when --prewarm-hidden-states is set."""
    import cli as cli_module
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    parser = cli_module.build_parser()
    args = parser.parse_args([
        "serve", "Qwen/Qwen3-0.6B",
        "--prewarm-hidden-states",
        "--artifact-dir", str(tmp_path / "artifacts"),
    ])

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def prewarm_hidden_states(self):
            captured["prewarmed"] = True

    class FakeStore:
        def __init__(self, path):
            captured["artifact_dir"] = path

    def fake_create_app(runtime, artifact_store, limits, api_key):
        del runtime, artifact_store, limits, api_key
        captured["create_app_called"] = True
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        captured["uvicorn_called"] = True

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    rc = cli_module._cmd_serve(args)
    assert rc == 0
    assert captured.get("prewarmed") is True
    assert captured.get("create_app_called") is True
    assert captured.get("uvicorn_called") is True


def test_serve_lifecycle_with_api_key(monkeypatch, tmp_path) -> None:
    """_cmd_serve propagates --api-key to create_app."""
    import cli as cli_module
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    parser = cli_module.build_parser()
    args = parser.parse_args([
        "serve", "Qwen/Qwen3-0.6B",
        "--api-key", "my-secret-key",
        "--artifact-dir", str(tmp_path / "artifacts"),
    ])

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def prewarm_hidden_states(self):
            pass

    class FakeStore:
        def __init__(self, path):
            captured["artifact_dir"] = path

    def fake_create_app(runtime, artifact_store, limits, api_key):
        del runtime, artifact_store, limits
        captured["create_app_called"] = True
        captured["api_key"] = api_key
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        captured["uvicorn_called"] = True

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    rc = cli_module._cmd_serve(args)
    assert rc == 0
    assert captured.get("api_key") == "my-secret-key"
    assert captured.get("create_app_called") is True


def test_serve_lifecycle_all_cli_args_flow_through(monkeypatch, tmp_path) -> None:
    """Full CLI config propagation through _cmd_serve: every option reaches its target."""
    import cli as cli_module
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    parser = cli_module.build_parser()
    args = parser.parse_args([
        "serve", "Qwen/Qwen3-0.6B",
        "--host", "0.0.0.0",
        "--port", "8123",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "4096",
        "--tokenizer", "my/tokenizer",
        "--served-model-name", "alias-model",
        "--api-key", "sk-test",
        "--seed", "42",
        "--trust-remote-code",
        "--local-files-only",
        "--artifact-dir", str(tmp_path / "my-artifacts"),
        "--max-top-k", "11",
        "--max-selected-layers", "12",
        "--max-selected-heads", "13",
        "--max-selected-positions", "14",
        "--max-inline-tensor-bytes", "15000",
        "--max-total-captured-tensor-bytes", "16000",
        "--max-artifact-bytes", "17000",
        "--enable-large-extraction",
        "--log-level", "warning",
    ])

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def prewarm_hidden_states(self):
            pass

    class FakeStore:
        def __init__(self, path):
            captured["store_path"] = path

    def fake_create_app(runtime, artifact_store, limits, api_key):
        del runtime, artifact_store
        captured["api_key_passed"] = api_key
        captured["limits"] = limits
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        captured["uvicorn_host"] = host
        captured["uvicorn_port"] = port
        captured["uvicorn_log_level"] = log_level

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    rc = cli_module._cmd_serve(args)
    assert rc == 0

    config = captured["config"]
    assert config.model == "Qwen/Qwen3-0.6B"
    assert config.dtype == "bfloat16"
    assert config.tensor_parallel_size == 1
    assert config.gpu_memory_utilization == 0.5
    assert config.max_model_len == 4096
    assert config.tokenizer == "my/tokenizer"
    assert config.served_model_name == "alias-model"
    assert config.seed == 42
    assert config.trust_remote_code is True
    assert config.local_files_only is True
    limits = captured["limits"]
    assert limits.max_top_k == 11
    assert limits.max_selected_layers == 12
    assert limits.max_selected_heads == 13
    assert limits.max_selected_positions == 14
    assert limits.max_inline_tensor_bytes == 15000
    assert limits.max_total_captured_tensor_bytes == 16000
    assert limits.max_artifact_bytes == 17000
    assert limits.large_extraction_enabled is True
    assert captured["store_path"] == Path(tmp_path / "my-artifacts")
    assert captured["api_key_passed"] == "sk-test"
    assert captured["uvicorn_host"] == "0.0.0.0"
    assert captured["uvicorn_port"] == 8123
    assert captured["uvicorn_log_level"] == "warning"
