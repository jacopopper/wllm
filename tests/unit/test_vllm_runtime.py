from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import pytest

from extractors.planning import RuntimeTopology
import runtime.vllm_compat as vllm_compat_module
from runtime.vllm_compat import (
    OnlineHiddenStateSelection,
    capture_online_hidden_states,
    capture_pooling_hidden_states,
    extract_attention_backend,
    extract_model_topology,
    extract_supported_runner_types,
    import_vllm,
    pooling_runner_environment,
    pooling_runner_kwargs,
)
import runtime.vllm_runtime as vllm_runtime_module
from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
from schemas.extraction import ExtractRequest
from schemas.openai import CompletionRequest
from server.errors import InvalidRequestError, ResourceLimitError, RuntimeUnavailableError, UnsupportedExtractionError


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeImports:
    SamplingParams = FakeSamplingParams
    version = "0.9.0"


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        return " ".join(m.get("content", "") for m in messages)

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return f"<token:{token_ids[-1]}>"


class CountingBatchTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        self.decode_calls: list[list[int]] = []
        self.batch_decode_calls: list[list[list[int]]] = []

    def decode(self, token_ids: list[int]) -> str:
        self.decode_calls.append(token_ids)
        return f"decode:{token_ids[-1]}"

    def batch_decode(self, token_id_batches: list[list[int]], **kwargs: Any) -> list[str]:
        del kwargs
        self.batch_decode_calls.append(token_id_batches)
        return [f"batch:{token_ids[-1]}" for token_ids in token_id_batches]


class EncodingFakeTokenizer(FakeTokenizer):
    @staticmethod
    def encode(prompt: str, **kwargs: Any) -> list[int]:
        del prompt, kwargs
        return [1]


class BosEncodingFakeTokenizer(FakeTokenizer):
    @staticmethod
    def encode(prompt: str, **kwargs: Any) -> list[int]:
        del prompt
        if kwargs.get("add_special_tokens") is False:
            return [101]
        return [1, 101]


class RestrictedSamplingParams:
    def __init__(self, max_tokens: int, temperature: float = 1.0) -> None:
        self.kwargs = {"max_tokens": max_tokens, "temperature": temperature}


class RestrictedImports:
    SamplingParams = RestrictedSamplingParams


class LogprobsOnlySamplingParams:
    def __init__(self, max_tokens: int, logprobs: int, temperature: float = 1.0) -> None:
        self.kwargs = {"max_tokens": max_tokens, "temperature": temperature, "logprobs": logprobs}


class LogprobsOnlyImports:
    SamplingParams = LogprobsOnlySamplingParams


class FakeCompletion:
    text = "ok"
    token_ids = [2]
    finish_reason = "stop"
    logprobs = None


class FakeOutput:
    prompt_token_ids = [1]
    prompt_logprobs = [{1: -0.4}]
    outputs = [FakeCompletion()]


class BosPromptOutput:
    prompt_token_ids = [1, 101]
    prompt_logprobs = [{1: -0.1}, {101: -0.2}]
    outputs = [FakeCompletion()]


class FakeLLM:
    def __init__(self) -> None:
        self.generate_calls = 0

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def generate(self, prompts: list[str], sampling: FakeSamplingParams) -> list[FakeOutput]:
        del prompts, sampling
        self.generate_calls += 1
        return [FakeOutput()]


class CountingTokenizerLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = CountingBatchTokenizer()
        self.get_tokenizer_calls = 0

    def get_tokenizer(self) -> CountingBatchTokenizer:
        self.get_tokenizer_calls += 1
        return self.tokenizer


class FakePoolingData:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class FakePoolingRequestOutput:
    def __init__(self, data: np.ndarray) -> None:
        self.outputs = FakePoolingData(data)


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
        self.hooks = []

    def register_forward_hook(self, hook: Any) -> FakeHookHandle:
        self.hooks.append(hook)
        return FakeHookHandle(self.hooks, hook)

    def emit(self, token_count: int) -> None:
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        output = (data + (self.index * 1000.0), None)
        for hook in list(self.hooks):
            hook(self, (), output)


class FakePoolingInnerModel:
    def __init__(self) -> None:
        self.layers = [FakeLayer(index) for index in range(12)]


class FakePoolingModel:
    def __init__(self) -> None:
        self.model = FakePoolingInnerModel()


class FakeOnlineLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakePoolingModel()

    def apply_model(self, func: Any) -> list[Any]:
        return [func(self.model)]

    def generate(self, prompts: list[str], sampling: FakeSamplingParams) -> list[FakeOutput]:
        del prompts, sampling
        self.generate_calls += 1
        for layer in self.model.model.layers:
            layer.emit(2)
        return [FakeOutput()]


class EncodingFakeOnlineLLM(FakeOnlineLLM):
    def get_tokenizer(self) -> EncodingFakeTokenizer:
        return EncodingFakeTokenizer()


class BosEncodingFakeOnlineLLM(FakeOnlineLLM):
    def get_tokenizer(self) -> BosEncodingFakeTokenizer:
        return BosEncodingFakeTokenizer()

    def generate(self, prompts: list[str], sampling: FakeSamplingParams) -> list[BosPromptOutput]:
        del prompts, sampling
        self.generate_calls += 1
        for layer in self.model.model.layers:
            layer.emit(2)
        return [BosPromptOutput()]


class GenerateFailsOnlineLLM(FakeOnlineLLM):
    def generate(self, prompts: list[str], sampling: FakeSamplingParams) -> list[FakeOutput]:
        del prompts, sampling
        self.generate_calls += 1
        for layer in self.model.model.layers:
            layer.emit(1)
        raise RuntimeError("generate failed")


class FakePoolingLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.encode_calls = []
        self.model = FakePoolingModel()

    def apply_model(self, func: Any) -> list[Any]:
        return [func(self.model)]

    def encode(self, *, prompt_token_ids: list[list[int]], use_tqdm: bool) -> list[FakePoolingRequestOutput]:
        self.encode_calls.append({"prompt_token_ids": prompt_token_ids, "use_tqdm": use_tqdm})
        token_count = len(prompt_token_ids[0])
        for layer in self.model.model.layers:
            layer.emit(token_count)
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        return [FakePoolingRequestOutput(data)]


class FakeModernPoolingLLM(FakePoolingLLM):
    supported_tasks = ("embed",)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.env_during_init = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")

    def encode(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        use_tqdm: bool,
        pooling_task: str,
    ) -> list[FakePoolingRequestOutput]:
        self.encode_calls.append({"prompts": prompts, "use_tqdm": use_tqdm, "pooling_task": pooling_task})
        token_count = len(prompts[0]["prompt_token_ids"])
        for layer in self.model.model.layers:
            layer.emit(token_count)
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        return [FakePoolingRequestOutput(data)]


class EncodeFailsPoolingLLM(FakePoolingLLM):
    def encode(self, *, prompt_token_ids: list[list[int]], use_tqdm: bool) -> list[FakePoolingRequestOutput]:
        self.encode_calls.append({"prompt_token_ids": prompt_token_ids, "use_tqdm": use_tqdm})
        raise RuntimeError("encode failed")


class EncodeAndCleanupFailPoolingLLM(EncodeFailsPoolingLLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.apply_model_calls = 0

    def apply_model(self, func: Any) -> list[Any]:
        self.apply_model_calls += 1
        if self.apply_model_calls == 2:
            raise RuntimeError("cleanup failed")
        return super().apply_model(func)


class ChunkedPoolingLLM(FakePoolingLLM):
    def encode(self, *, prompt_token_ids: list[list[int]], use_tqdm: bool) -> list[FakePoolingRequestOutput]:
        self.encode_calls.append({"prompt_token_ids": prompt_token_ids, "use_tqdm": use_tqdm})
        token_count = len(prompt_token_ids[0])
        first_chunk = max(token_count // 2, 1)
        second_chunk = token_count - first_chunk
        for layer in self.model.model.layers:
            layer.emit(first_chunk)
            if second_chunk:
                layer.emit(second_chunk)
        data = np.arange(token_count * 32, dtype=np.float32).reshape(token_count, 32)
        return [FakePoolingRequestOutput(data)]


class FakePoolingImports:
    module = object()
    version = "0.9.0"
    LLM = FakePoolingLLM
    SamplingParams = FakeSamplingParams


class FakeModernPoolingImports:
    module = object()
    version = "0.10.2"
    LLM = FakeModernPoolingLLM
    SamplingParams = FakeSamplingParams


class FakeOnlineImports:
    module = object()
    version = "0.10.2"
    LLM = FakeOnlineLLM
    SamplingParams = FakeSamplingParams


def test_import_vllm_rejects_unvalidated_versions(monkeypatch) -> None:
    monkeypatch.setattr(vllm_compat_module.importlib.metadata, "version", lambda _name: "0.10.3")

    try:
        import_vllm()
    except RuntimeUnavailableError as exc:
        assert exc.code == "unsupported_vllm_version"
        assert exc.details == {"supported": "0.10.2", "installed": "0.10.3"}
    else:
        raise AssertionError("unvalidated vLLM versions should be rejected")


def test_import_vllm_accepts_validated_version(monkeypatch) -> None:
    class FakeModule:
        LLM = object
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_compat_module.importlib.metadata, "version", lambda _name: "0.10.2")
    monkeypatch.setattr(vllm_compat_module.importlib, "import_module", lambda _name: FakeModule)

    imports = import_vllm()

    assert imports.version == "0.10.2"
    assert imports.LLM is object
    assert imports.SamplingParams is FakeSamplingParams


def test_logprobs_true_normalizes_to_one() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakeImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "logprobs": True})
    sampling = runtime._sampling_params(request)
    assert sampling.kwargs["logprobs"] == 1


def test_logprobs_integer_preserved() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakeImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "logprobs": 5})
    sampling = runtime._sampling_params(request)
    assert sampling.kwargs["logprobs"] == 5


def test_logprobs_false_is_disabled_even_when_sampling_params_lack_logprobs() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = RestrictedImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "logprobs": False})

    sampling = runtime._sampling_params(request)

    assert sampling.kwargs == {"max_tokens": 16, "temperature": 1.0}


def test_config_seed_is_used_when_request_omits_seed() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", seed=42))
    runtime._imports = FakeImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello"})

    sampling = runtime._sampling_params(request)

    assert sampling.kwargs["seed"] == 42


def test_config_seed_requires_sampling_params_support() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", seed=42))
    runtime._imports = RestrictedImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello"})

    try:
        runtime._sampling_params(request)
    except InvalidRequestError as exc:
        assert exc.code == "unsupported_sampling_parameter"
        assert exc.param == "seed"
    else:
        raise AssertionError("configured seed should require SamplingParams.seed support")


def test_explicit_unsupported_sampling_parameter_is_rejected() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = RestrictedImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "top_k": 20})

    try:
        runtime._sampling_params(request)
    except InvalidRequestError as exc:
        assert exc.code == "unsupported_sampling_parameter"
        assert exc.param == "top_k"
        assert exc.status_code == 422
    else:
        raise AssertionError("unsupported sampling parameter should fail")


def test_default_omitted_sampling_parameter_is_not_rejected() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = RestrictedImports()
    request = CompletionRequest.model_validate({"model": "fake", "prompt": "hello"})

    sampling = runtime._sampling_params(request)

    assert sampling.kwargs == {"max_tokens": 16, "temperature": 1.0}


def test_extraction_logprobs_require_sampling_support() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = RestrictedImports()
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"logprobs": {"top_k": 1}}}
    )

    try:
        runtime._sampling_params(request, force_logprobs=True)
    except UnsupportedExtractionError as exc:
        assert exc.status_code == 501
        assert exc.code == "token_logprobs_unavailable"
        assert exc.param == "extract.logprobs"
    else:
        raise AssertionError("extract.logprobs should require SamplingParams.logprobs")


def test_extraction_prompt_logprobs_set_sampling_params() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeImports()
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"logprobs": {"top_k": 3, "include_prompt": True}}}
    )

    sampling = runtime._sampling_params(request, force_logprobs=True)

    assert sampling.kwargs["logprobs"] == 3
    assert sampling.kwargs["prompt_logprobs"] == 3


def test_extraction_prompt_logprobs_require_sampling_support() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = LogprobsOnlyImports()
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"logprobs": {"top_k": 1, "include_prompt": True}}}
    )

    try:
        runtime._sampling_params(request, force_logprobs=True)
    except UnsupportedExtractionError as exc:
        assert exc.status_code == 501
        assert exc.code == "prompt_logprobs_unavailable"
        assert exc.param == "extract.logprobs.include_prompt"
    else:
        raise AssertionError("prompt logprob extraction should require SamplingParams.prompt_logprobs")


def test_normal_completion_does_not_allocate_extraction_collectors() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    assert runtime._collector_registry is None
    response = runtime.generate_completion(
        CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "max_tokens": 1})
    )

    assert response["choices"][0]["text"] == "ok"
    assert runtime._llm.generate_calls == 1
    assert runtime._collector_registry is None


def test_lazy_vllm_initialization_is_serialized(monkeypatch) -> None:
    class LoadableLLM:
        calls = 0
        lock = threading.Lock()

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            with self.lock:
                type(self).calls += 1
            time.sleep(0.01)

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    errors = []

    def load() -> None:
        try:
            runtime._ensure_loaded()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert LoadableLLM.calls == 1


def test_lazy_initialization_does_not_publish_llm_before_metadata(monkeypatch) -> None:
    class LoadableLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    observed = []

    def extract_topology(llm: Any) -> RuntimeTopology | None:
        assert isinstance(llm, LoadableLLM)
        observed.append(runtime._llm is None)
        return None

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    monkeypatch.setattr(vllm_runtime_module, "extract_model_topology", extract_topology)

    runtime._ensure_loaded()

    assert observed == [True]
    assert isinstance(runtime._llm, LoadableLLM)


def test_online_hidden_state_config_uses_eager_inprocess_generation_runner(monkeypatch) -> None:
    class LoadableLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.env_during_init = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))

    runtime._ensure_loaded()

    assert runtime._llm.kwargs["enforce_eager"] is True
    assert runtime._llm.kwargs["enable_prefix_caching"] is False
    assert runtime._llm.env_during_init == "0"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert runtime.capabilities().hidden_states.details["online_hidden_states_enabled"] is True
    assert "online" in runtime.capabilities().hidden_states.details["capture_modes"]


def test_prewarm_generation_runner_uses_eager_inprocess_for_tokenizer_compat(monkeypatch) -> None:
    class LoadableLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.env_during_init = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))

    runtime._prewarm_initializing = True
    try:
        runtime._ensure_loaded()
    finally:
        runtime._prewarm_initializing = False

    assert runtime._llm.kwargs["enforce_eager"] is True
    assert runtime._llm.kwargs["enable_prefix_caching"] is False
    assert runtime._llm.env_during_init == "0"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"


def test_failed_initialization_metadata_does_not_publish_partial_llm(monkeypatch) -> None:
    class LoadableLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    monkeypatch.setattr(
        vllm_runtime_module,
        "extract_model_topology",
        lambda _llm: (
            _ for _ in ()).throw(
            RuntimeError("metadata failed")))
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))

    try:
        runtime._ensure_loaded()
    except RuntimeUnavailableError as exc:
        assert exc.code == "vllm_initialization_failed"
        assert "metadata failed" in exc.details["exception"]
    else:
        raise AssertionError("metadata initialization failure should be structured")

    assert runtime._llm is None
    assert runtime._capabilities is None


def test_pooling_llm_is_not_published_until_encode_surface_is_verified() -> None:
    class MissingEncodeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        version = "0.10.2"
        LLM = MissingEncodeLLM

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = Imports()
    runtime._supported_runner_types = ["generate", "pooling"]

    try:
        runtime._ensure_pooling_llm()
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
    else:
        raise AssertionError("pooling runner without encode should be rejected")

    assert runtime._pooling_llm is None


def test_prewarm_hidden_states_initializes_pooling_runner_without_capture() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4, prewarm_hidden_states=True))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    runtime.prewarm_hidden_states()

    assert runtime._pooling_llm is not None
    assert runtime._pooling_llm.kwargs["task"] == "embed"
    assert runtime._pooling_llm.kwargs["enable_prefix_caching"] is False
    assert runtime._pooling_llm.encode_calls == []


def test_prewarm_hidden_states_rejects_unsupported_configuration_before_pooling() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.9, prewarm_hidden_states=True))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    try:
        runtime.prewarm_hidden_states()
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
        assert exc.details["capability"]["details"]["gpu_memory_utilization"] == 0.9
    else:
        raise AssertionError("prewarm should reject unsupported hidden-state configurations")

    assert runtime._pooling_llm is None


def test_normal_chat_does_not_allocate_extraction_collectors() -> None:
    from schemas.openai import ChatCompletionRequest

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    assert runtime._collector_registry is None
    response = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        )
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert runtime._collector_registry is None


def test_normal_chat_does_not_build_extraction_plans(monkeypatch) -> None:
    from schemas.openai import ChatCompletionRequest

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=32, hidden_size=32)

    plan_calls: list[Any] = []

    def _boom_plan(*args: Any, **kwargs: Any) -> Any:
        plan_calls.append((args, kwargs))
        raise AssertionError("compile_extraction_plan must not be called for normal chat completion")

    monkeypatch.setattr(vllm_runtime_module, "compile_extraction_plan", _boom_plan)

    response = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        )
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert plan_calls == []


def test_normal_completion_does_not_build_extraction_plans(monkeypatch) -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=32, hidden_size=32)

    plan_calls: list[Any] = []

    def _boom_plan(*args: Any, **kwargs: Any) -> Any:
        plan_calls.append((args, kwargs))
        raise AssertionError("compile_extraction_plan must not be called for normal completion")

    monkeypatch.setattr(vllm_runtime_module, "compile_extraction_plan", _boom_plan)

    response = runtime.generate_completion(
        CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "max_tokens": 1})
    )

    assert response["choices"][0]["text"] == "ok"
    assert plan_calls == []


def test_normal_chat_does_not_write_artifacts(tmp_path, monkeypatch) -> None:
    from schemas.openai import ChatCompletionRequest

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    artifact_calls: list[str] = []

    class _SentinelStore:
        def put(self, *args: Any, **kwargs: Any) -> Any:
            artifact_calls.append("put")
            raise AssertionError("ArtifactStore.put must not be called for normal chat completion")

        def put_trace_bundle(self, *args: Any, **kwargs: Any) -> Any:
            artifact_calls.append("put_trace_bundle")
            raise AssertionError("ArtifactStore.put_trace_bundle must not be called for normal chat completion")

    monkeypatch.setattr(vllm_runtime_module, "ArtifactStore", _SentinelStore)

    response = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        )
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert artifact_calls == []
    # The configured artifact directory must remain untouched (no files created).
    assert list(tmp_path.iterdir()) == []


def test_normal_chat_does_not_register_hooks(monkeypatch) -> None:
    import torch

    from schemas.openai import ChatCompletionRequest

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    hook_calls: list[Any] = []
    original_register = torch.nn.Module.register_forward_hook

    def _recording_register_hook(self_module: Any, hook: Any, *args: Any, **kwargs: Any) -> Any:
        hook_calls.append((self_module, hook))
        return original_register(self_module, hook, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "register_forward_hook", _recording_register_hook)
    # Guard the vLLM-compat hook-installation entrypoints as well; none should fire.
    online_calls: list[Any] = []
    pooling_calls: list[Any] = []
    monkeypatch.setattr(
        vllm_compat_module,
        "capture_online_hidden_states",
        lambda *a, **k: online_calls.append((a, k)) or (_ for _ in ()).throw(
            AssertionError("capture_online_hidden_states must not be called for normal chat completion")
        ),
    )
    monkeypatch.setattr(
        vllm_compat_module,
        "capture_pooling_hidden_states",
        lambda *a, **k: pooling_calls.append((a, k)) or (_ for _ in ()).throw(
            AssertionError("capture_pooling_hidden_states must not be called for normal chat completion")
        ),
    )

    response = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        )
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert hook_calls == []
    assert online_calls == []
    assert pooling_calls == []


def test_normal_chat_does_not_activate_trace_context(monkeypatch) -> None:
    from schemas.openai import ChatCompletionRequest
    from tracing.context import active_trace_id

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    # Ensure the context variable is clean before the call.
    active_trace_id.set(None)

    trace_ids_seen_during_generate: list[Any] = []
    original_generate = runtime._llm.generate

    def _spying_generate(prompts: list[str], sampling: Any) -> list[Any]:
        trace_ids_seen_during_generate.append(active_trace_id.get())
        return original_generate(prompts, sampling)

    monkeypatch.setattr(runtime._llm, "generate", _spying_generate)

    assert active_trace_id.get() is None
    response = runtime.generate_chat(
        ChatCompletionRequest.model_validate(
            {"model": "fake", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        )
    )
    assert active_trace_id.get() is None

    assert response["choices"][0]["message"]["content"] == "ok"
    # Inside the vLLM generate call (the only place hooks/collectors could fire),
    # no trace context must be active.
    assert trace_ids_seen_during_generate == [None]


def test_extract_with_empty_spec_creates_no_extraction_overhead(monkeypatch, tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from schemas.extraction import ExtractRequest

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=32, hidden_size=32)

    plan_calls: list[Any] = []
    original_compile = vllm_runtime_module.compile_extraction_plan

    def _spying_compile(spec: Any, *args: Any, **kwargs: Any) -> Any:
        plan = original_compile(spec, *args, **kwargs)
        plan_calls.append(plan)
        return plan

    monkeypatch.setattr(vllm_runtime_module, "compile_extraction_plan", _spying_compile)

    # No hooks may be installed and no hidden-state capture machinery may run.
    online_calls: list[Any] = []
    pooling_calls: list[Any] = []
    monkeypatch.setattr(
        vllm_compat_module,
        "capture_online_hidden_states",
        lambda *a, **k: online_calls.append((a, k)) or (_ for _ in ()).throw(
            AssertionError("capture_online_hidden_states must not be called for an empty extract spec")
        ),
    )
    monkeypatch.setattr(
        vllm_compat_module,
        "capture_pooling_hidden_states",
        lambda *a, **k: pooling_calls.append((a, k)) or (_ for _ in ()).throw(
            AssertionError("capture_pooling_hidden_states must not be called for an empty extract spec")
        ),
    )

    # Capture-mode resolution must report None (no hidden-state capture requested).
    request = ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {}})
    assert runtime._hidden_state_capture_mode(request) is None

    store = ArtifactStore(tmp_path / "artifacts")
    trace = runtime.generate_extract(
        request,
        limits=ResourceLimits(),
        artifact_store=store,
        persist=False,
    )

    assert trace.schema_version == "wllm.trace.v1"
    # Empty extract spec produces an empty plan (no logprobs, hidden states, or attentions).
    assert plan_calls, "compile_extraction_plan should run once to produce the empty plan"
    plan = plan_calls[-1]
    assert plan.logprobs is None
    assert plan.hidden_states == []
    assert plan.attentions == []
    # No artifacts written and no inline extraction payload.
    assert trace.artifacts == []
    assert trace.trace.tokens.token_ids == []
    assert trace.trace.tokens.tokens == []
    assert trace.trace.logprobs == {}
    assert trace.trace.hidden_states == []
    # No hook-installation entrypoint fired.
    assert online_calls == []
    assert pooling_calls == []
    # Artifact directory must not be created by an empty-spec extraction.
    assert not (tmp_path / "artifacts").exists()


def test_extract_creates_collector_registry_and_cleans_up() -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    assert runtime._collector_registry is None
    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {"tokens": True}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
        persist=False,
    )
    assert trace.schema_version == "wllm.trace.v1"
    assert runtime._collector_registry is not None
    assert runtime._collector_registry.active_ids() == set()


def test_generate_extract_uses_batch_decode_when_tokens_requested(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    llm = CountingTokenizerLLM()
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = llm

    trace = runtime.generate_extract(
        ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {"tokens": True}}),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.trace.tokens.tokens == ["batch:1", "batch:2"]
    assert llm.tokenizer.batch_decode_calls == [[[1], [2]]]
    assert llm.tokenizer.decode_calls == []


def test_attention_extract_requires_runtime_opt_in(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=2, num_attention_heads=2, hidden_size=32)

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "attentions": [
                            {
                                "layers": 0,
                                "heads": 0,
                                "query_positions": "last_generated",
                                "key_positions": "previous_token",
                            }
                        ]
                    },
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "attention_weights_unavailable"
    assert exc.value.param == "extract.attentions"
    assert runtime._llm.generate_calls == 0


def test_generate_extract_replays_attention_weights_when_requested(tmp_path, monkeypatch) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    replay = object()
    load_calls: list[dict[str, Any]] = []
    capture_calls: list[list[int]] = []

    def fake_load(**kwargs: Any) -> object:
        load_calls.append(kwargs)
        return replay

    def fake_capture(replay_arg: object, *, token_ids: list[int]) -> dict[int, np.ndarray]:
        assert replay_arg is replay
        capture_calls.append(token_ids)
        return {0: np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)}

    monkeypatch.setattr(vllm_runtime_module, "load_transformers_attention_replay", fake_load)
    monkeypatch.setattr(vllm_runtime_module, "capture_transformers_replay_attentions", fake_capture)
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_attention_weights=True, local_files_only=True))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=2, num_attention_heads=2, hidden_size=32)

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {
                "model": "fake",
                "prompt": "hello",
                "extract": {
                    "attentions": [
                        {
                            "layers": 0,
                            "heads": 0,
                            "query_positions": "last_generated",
                            "key_positions": "previous_token",
                        }
                    ]
                },
            }
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert load_calls == [
        {
            "model": "fake",
            "tokenizer": None,
            "dtype": "auto",
            "trust_remote_code": False,
            "local_files_only": True,
        }
    ]
    assert capture_calls == [[1, 2]]
    assert runtime._llm.generate_calls == 1
    record = trace.trace.attentions[0]
    assert record.shape == [1, 1, 1, 1]
    assert record.data[0][0][0][0] == 2.0
    assert trace.metadata.capture["attentions"]["capture_metadata"]["backend"] == "transformers_replay"


def test_normal_completion_bypasses_attention_replay_when_enabled(monkeypatch) -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_attention_weights=True))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    monkeypatch.setattr(
        vllm_runtime_module,
        "load_transformers_attention_replay",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("attention replay must not load")),
    )
    monkeypatch.setattr(
        vllm_runtime_module,
        "capture_transformers_replay_attentions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("attention replay must not run")),
    )

    response = runtime.generate_completion(
        CompletionRequest.model_validate({"model": "fake", "prompt": "hello", "max_tokens": 1})
    )

    assert response["choices"][0]["text"] == "ok"
    assert runtime._llm.generate_calls == 1
    assert runtime._attention_replay is None


def test_attention_replay_memory_limit_is_checked_before_loading_replay(tmp_path, monkeypatch) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_attention_weights=True))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=2, num_attention_heads=2, hidden_size=32)
    monkeypatch.setattr(
        vllm_runtime_module,
        "load_transformers_attention_replay",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("replay should not load over limit")),
    )

    with pytest.raises(ResourceLimitError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "attentions": [
                            {
                                "layers": 0,
                                "heads": 0,
                                "query_positions": "last_generated",
                                "key_positions": "previous_token",
                            }
                        ]
                    },
                }
            ),
            limits=ResourceLimits(max_total_captured_tensor_bytes=31, max_inline_tensor_bytes=1024),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.param == "extract.attentions"
    assert exc.value.details["estimated_raw_capture_bytes"] == 64


def test_hidden_only_extract_skips_token_decoding(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    llm = CountingTokenizerLLM()
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakePoolingImports()
    runtime._llm = llm
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {
                "hidden_states": [{"layers": -1, "positions": "last_generated"}]}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.trace.tokens.tokens == []
    assert trace.trace.hidden_states
    assert llm.get_tokenizer_calls == 0
    assert llm.tokenizer.batch_decode_calls == []
    assert llm.tokenizer.decode_calls == []


def test_online_hidden_extract_requires_runtime_opt_in(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeOnlineImports()
    runtime._llm = FakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)

    try:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {"hidden_states": [{"layers": 5, "positions": "prompt", "capture_mode": "online"}]},
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except UnsupportedExtractionError as exc:
        assert exc.code == "online_hidden_states_disabled"
        assert exc.param == "extract.hidden_states.capture_mode"
    else:
        raise AssertionError("online capture should require explicit runtime opt-in")


def test_online_hidden_extract_uses_generation_runner_hooks(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    llm = FakeOnlineLLM()
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = llm
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {
                "model": "fake",
                "prompt": "hello",
                "extract": {"hidden_states": [{"layers": 5, "positions": "prompt", "capture_mode": "online"}]},
            }
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert llm.generate_calls == 1
    assert record.capture_mode == "online"
    assert record.capture_phase == "prompt_prefill"
    assert record.capture_metadata["hook_scope"] == "active_generation_runner"
    assert record.data[0][0][:3] == [5000.0, 5001.0, 5002.0]
    assert trace.metadata.capture["hidden_states"]["capture_modes"] == ["online"]
    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_online_hidden_extract_rejects_raw_capture_over_limit_before_generation(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    llm = FakeOnlineLLM()
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = llm
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)

    with pytest.raises(ResourceLimitError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "max_tokens": 2,
                    "extract": {"hidden_states": [{"layers": 5, "positions": "prompt", "capture_mode": "online"}]},
                }
            ),
            limits=ResourceLimits(max_total_captured_tensor_bytes=100),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.details["captured_layers"] == 1
    assert exc.value.details["captured_tokens_per_layer"] == 7
    assert llm.generate_calls == 0
    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_extract_supports_multiple_samples_for_ergonomics(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "n": 2, "extract": {"tokens": True}}
    )

    # multi-sample now allowed (for UQ like semantic entropy etc); internals may be limited to first
    # we don't assert full success without fake model, but no early reject
    try:
        runtime.generate_extract(
            request,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except Exception as exc:
        # may fail later on fake, but not on the n check
        assert "unsupported_extraction_sample_count" not in str(exc)
    else:
        raise AssertionError("multi-sample extraction requests should be rejected")


def test_generate_extract_includes_prompt_logprobs_when_requested() -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))
    runtime._imports = FakeImports()
    runtime._llm = FakeLLM()

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {"logprobs": {"top_k": 1, "include_prompt": True}}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(__import__("pathlib").Path("/tmp/wllm-test-artifacts")),
        persist=False,
    )

    assert trace.trace.logprobs["prompt"][0]["top_logprobs"][0]["token_id"] == 1
    assert trace.trace.logprobs["prompt"][0]["top_logprobs"][0]["logprob"] == -0.4


def test_extract_model_topology_from_fake_vllm_config() -> None:
    class HFConfig:
        num_hidden_layers = 24
        num_attention_heads = 16
        hidden_size = 2048

    class ModelConfig:
        hf_config = HFConfig()

    class Engine:
        model_config = ModelConfig()

    class LLM:
        llm_engine = Engine()

    topology = extract_model_topology(LLM())
    assert topology is not None
    assert topology.num_layers == 24
    assert topology.num_attention_heads == 16
    assert topology.hidden_size == 2048


def test_extract_attention_backend_prefers_env(monkeypatch) -> None:
    class Config:
        attention_backend = "FLASH_ATTN"

    class Engine:
        model_config = Config()

    class LLM:
        llm_engine = Engine()

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "XFORMERS")

    assert extract_attention_backend(LLM()) == "XFORMERS"


def test_extract_attention_backend_from_fake_vllm_config(monkeypatch) -> None:
    class Config:
        attention_backend = "FLASH_ATTN"

    class Engine:
        model_config = Config()

    class LLM:
        llm_engine = Engine()

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_USE_FLASHINFER", raising=False)

    assert extract_attention_backend(LLM()) == "FLASH_ATTN"


def test_extract_supported_runner_types_from_fake_vllm_config() -> None:
    class Config:
        supported_runner_types = ["generate", "pooling"]

    class Engine:
        model_config = Config()

    class LLM:
        llm_engine = Engine()

    assert extract_supported_runner_types(LLM()) == ["generate", "pooling"]


def test_pooling_runner_kwargs_follow_vllm_version() -> None:
    assert pooling_runner_kwargs("0.9.2")["task"] == "embed"
    assert pooling_runner_kwargs("0.9.2")["enforce_eager"] is True
    assert pooling_runner_kwargs("0.9.2")["enable_prefix_caching"] is False
    modern = pooling_runner_kwargs("0.10.2")
    assert modern["runner"] == "pooling"
    assert modern["convert"] == "embed"
    assert modern["enforce_eager"] is True
    assert modern["enable_prefix_caching"] is False
    assert modern["override_pooler_config"] == {"pooling_type": "LAST", "normalize": False}


def test_pooling_runner_environment_disables_v1_multiprocessing_for_modern_vllm() -> None:
    assert pooling_runner_environment("0.9.2") == {}
    assert pooling_runner_environment("0.10.2") == {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}


def test_capture_pooling_hidden_states_removes_hooks_after_encode_failure() -> None:
    pooling_llm = EncodeFailsPoolingLLM()

    try:
        capture_pooling_hidden_states(pooling_llm, token_ids=[1, 2], layers=[0, 3])
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
        assert "encode failed" in exc.details["exception"]
    else:
        raise AssertionError("failing pooling encode should be reported as unsupported extraction")

    assert all(not layer.hooks for layer in pooling_llm.model.model.layers)


def test_capture_pooling_hidden_states_preserves_encode_error_when_cleanup_fails() -> None:
    pooling_llm = EncodeAndCleanupFailPoolingLLM()

    try:
        capture_pooling_hidden_states(pooling_llm, token_ids=[1, 2], layers=[0])
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
        assert "encode failed" in exc.details["exception"]
        assert "cleanup failed" in exc.details["cleanup_exception"]
    else:
        raise AssertionError("failing pooling encode should remain the primary error")


def test_capture_pooling_hidden_states_concatenates_chunked_forward_outputs() -> None:
    pooling_llm = ChunkedPoolingLLM()

    captures = capture_pooling_hidden_states(pooling_llm, token_ids=[1, 2, 3, 4, 5], layers=[0, 3])

    assert captures[0].shape == (5, 32)
    assert captures[3].shape == (5, 32)
    assert captures[0][0, 0] == 0.0
    assert captures[0][2, 0] == 0.0
    assert captures[0][4, 0] == 64.0
    assert captures[3][4, 0] == 3064.0
    assert all(not layer.hooks for layer in pooling_llm.model.model.layers)


def test_capture_online_hidden_states_scopes_hooks_around_generate() -> None:
    llm = FakeOnlineLLM()
    observed_hooks = []

    def generate() -> str:
        observed_hooks.append([len(layer.hooks) for layer in llm.model.model.layers[:4]])
        for layer in llm.model.model.layers:
            layer.emit(2)
        return "ok"

    capture = capture_online_hidden_states(llm, layers=[0, 3], generate=generate)

    assert capture.output == "ok"
    assert observed_hooks == [[1, 0, 0, 1]]
    assert capture.tensors[0].shape == (2, 32)
    assert capture.tensors[3][1, 0] == 3032.0
    assert capture.capture_phase == "prompt_prefill_decode_best_effort"
    assert capture.metadata["layer_chunk_shapes"]["0"] == [[2, 32]]
    assert capture.metadata["capture_filter"] is None
    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_capture_online_hidden_states_can_slice_dense_prefix_before_copy() -> None:
    llm = FakeOnlineLLM()

    def generate() -> str:
        for layer in llm.model.model.layers:
            layer.emit(2)
            layer.emit(1)
        return "ok"

    capture = capture_online_hidden_states(llm, layers=[0, 3], generate=generate, capture_max_position=0)

    assert capture.output == "ok"
    assert capture.tensors[0].shape == (1, 32)
    assert capture.tensors[3].shape == (1, 32)
    assert capture.tensors[3][0, 0] == 3000.0
    assert capture.metadata["layer_chunk_shapes"]["0"] == [[1, 32]]
    assert capture.metadata["capture_filter"] == {
        "type": "dense_prefix",
        "max_source_position": 0,
        "captured_source_positions": [0, 0],
    }
    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_capture_online_hidden_states_can_select_and_pool_on_worker() -> None:
    llm = FakeOnlineLLM()

    def generate() -> str:
        for layer in llm.model.model.layers:
            layer.emit(3)
        return "ok"

    capture = capture_online_hidden_states(
        llm,
        layers=[0, 3],
        generate=generate,
        select_hidden_states=lambda _output: [
            OnlineHiddenStateSelection(
                name="hidden_states_0",
                layers=[0, 3],
                positions=[1, 2],
                pool="mean",
            ),
            OnlineHiddenStateSelection(
                name="hidden_states_1",
                layers=[3],
                positions=[2],
                pool=None,
            ),
        ],
    )

    assert capture.output == "ok"
    assert capture.tensors == {}
    assert capture.selected_tensors["hidden_states_0"].shape == (2, 32)
    assert capture.selected_tensors["hidden_states_0"][0, 0] == 48.0
    assert capture.selected_tensors["hidden_states_0"][1, 0] == 3048.0
    assert capture.selected_tensors["hidden_states_1"].shape == (1, 1, 32)
    assert capture.selected_tensors["hidden_states_1"][0, 0, 0] == 3064.0
    assert capture.metadata["selection_mode"] == "worker_selected"
    assert capture.metadata["selected_tensor_shapes"] == {
        "hidden_states_0": [2, 32],
        "hidden_states_1": [1, 1, 32],
    }
    assert capture.metadata["layer_chunk_shapes"]["0"] == [[3, 32]]
    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_capture_online_hidden_states_removes_hooks_after_generation_failure() -> None:
    llm = FakeOnlineLLM()

    def generate() -> None:
        for layer in llm.model.model.layers:
            layer.emit(1)
        raise RuntimeError("generate failed")

    with pytest.raises(RuntimeError, match="generate failed"):
        capture_online_hidden_states(llm, layers=[0, 3], generate=generate)

    assert all(not layer.hooks for layer in llm.model.model.layers)


def test_generate_extract_uses_scoped_pooling_hooks_for_hidden_states(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {
                "hidden_states": [{"layers": -1, "positions": "last_generated"}]}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert runtime._pooling_llm.kwargs["task"] == "embed"
    assert runtime._pooling_llm.kwargs["enable_prefix_caching"] is False
    assert runtime._pooling_llm.kwargs["override_pooler_config"]["pooling_type"] == "ALL"
    assert runtime._pooling_llm.encode_calls == [{"prompt_token_ids": [[1, 2]], "use_tqdm": False}]
    record = trace.trace.hidden_states[0]
    assert record.layers == [11]
    assert record.positions == [1]
    assert record.capture_site in ("block", "transformer_block_output")
    assert record.shape == [1, 1, 32]
    assert record.data[0][0][:3] == [11032.0, 11033.0, 11034.0]
    assert all(not layer.hooks for layer in runtime._pooling_llm.model.model.layers)


def test_generate_extract_uses_online_hooks_for_hidden_states(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = FakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {
                "model": "fake",
                "prompt": "hello",
                "extract": {
                    "hidden_states": [
                        {"layers": -1, "positions": "last_generated", "capture_mode": "online"}
                    ]
                },
            }
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert runtime._pooling_llm is None
    assert runtime._llm.generate_calls == 1
    record = trace.trace.hidden_states[0]
    assert record.layers == [11]
    assert record.positions == [1]
    assert record.capture_mode == "online"
    assert record.capture_phase == "decode"
    assert record.capture_site in ("block", "transformer_block_output")
    assert record.position_semantics["decoder_only_prediction_semantics"]
    assert record.position_semantics["source_positions"] == [0]
    assert record.capture_metadata["hook_scope"] == "active_generation_runner"
    assert record.capture_metadata["selection_mode"] == "worker_selected"
    assert record.capture_metadata["selected_tensor_shapes"] == {"hidden_states_0": [1, 1, 32]}
    assert record.shape == [1, 1, 32]
    assert record.data[0][0][:3] == [11000.0, 11001.0, 11002.0]
    assert trace.metadata.resolved_selectors["hidden_states"][0]["capture_mode"] == "online"
    assert trace.metadata.capture["hidden_states"]["capture_modes"] == ["online"]
    assert all(not layer.hooks for layer in runtime._llm.model.model.layers)


def test_generate_extract_slices_online_prompt_capture_prefix(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = EncodingFakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

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
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert runtime._pooling_llm is None
    assert runtime._llm.generate_calls == 1
    assert record.capture_mode == "online"
    assert record.capture_phase == "prompt_prefill"
    assert record.positions == [0]
    assert record.shape == [1, 1, 32]
    assert record.data[0][0][:3] == [5000.0, 5001.0, 5002.0]
    assert record.capture_metadata["layer_chunk_shapes"]["5"] == [[1, 32]]
    assert record.capture_metadata["capture_filter"] == {
        "type": "dense_prefix",
        "max_source_position": 0,
        "captured_source_positions": [0, 0],
    }
    assert all(not layer.hooks for layer in runtime._llm.model.model.layers)


def test_generate_extract_online_prompt_capture_counts_special_tokens(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = BosEncodingFakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

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
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert trace.generation["usage"]["prompt_tokens"] == 2
    assert record.positions == [0, 1]
    assert record.shape == [1, 2, 32]
    assert record.capture_metadata["layer_chunk_shapes"]["5"] == [[2, 32]]
    assert record.capture_metadata["capture_filter"] == {
        "type": "dense_prefix",
        "max_source_position": 1,
        "captured_source_positions": [0, 1],
    }


def test_generate_extract_uses_modern_pooling_runner_for_hidden_states(tmp_path, monkeypatch) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakeModernPoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {
                "hidden_states": [{"layers": -1, "positions": "last_generated"}]}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert runtime._pooling_llm.kwargs["runner"] == "pooling"
    assert runtime._pooling_llm.kwargs["convert"] == "embed"
    assert runtime._pooling_llm.kwargs["enforce_eager"] is True
    assert runtime._pooling_llm.kwargs["enable_prefix_caching"] is False
    assert runtime._pooling_llm.kwargs["override_pooler_config"]["pooling_type"] == "LAST"
    assert runtime._pooling_llm.env_during_init == "0"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert runtime._pooling_llm.encode_calls == [
        {"prompts": [{"prompt_token_ids": [1, 2]}], "use_tqdm": False, "pooling_task": "embed"}
    ]
    assert trace.trace.hidden_states[0].data[0][0][:3] == [11032.0, 11033.0, 11034.0]


def test_generate_extract_rejects_raw_hidden_capture_over_limit_before_pooling(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    try:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {
                    "hidden_states": [{"layers": -1, "positions": "last_generated"}]}}
            ),
            limits=ResourceLimits(max_total_captured_tensor_bytes=255, max_inline_tensor_bytes=10_000),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except ResourceLimitError as exc:
        assert exc.param == "extract.hidden_states"
        assert exc.details["estimated_raw_capture_bytes"] == 256
        assert runtime._pooling_llm is None
    else:
        raise AssertionError("raw hidden-state capture over the hard limit should be rejected before pooling")


def test_generate_extract_supports_middle_layer_with_scoped_pooling_hooks(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": "fake", "prompt": "hello", "extract": {
                "hidden_states": [{"layers": "middle", "positions": "last"}]}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert record.layers == [5]
    assert record.positions == [1]
    assert record.data[0][0][:3] == [5032.0, 5033.0, 5034.0]


def test_generate_extract_rejects_hidden_states_with_tensor_parallelism(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", tensor_parallel_size=2, enable_online_hidden_states=True))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    try:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {
                    "hidden_states": [{"layers": "middle", "positions": "last"}]}}
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
        assert exc.details["capability"]["details"]["tensor_parallel_size"] == 2
        assert runtime._pooling_llm is None
    else:
        raise AssertionError("tensor-parallel hidden-state capture should be rejected")


def test_generate_extract_rejects_online_hidden_states_with_tensor_parallelism(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", tensor_parallel_size=2, enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = FakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "hidden_states": [
                            {"layers": "middle", "positions": "last", "capture_mode": "online"}
                        ]
                    },
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "online_hidden_state_parallelism_unavailable"
    assert exc.value.details["tensor_parallel_size"] == 2
    assert runtime._llm.generate_calls == 0


def test_generate_extract_rejects_online_hidden_states_for_unsupported_vllm_version(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    class OldOnlineImports(FakeOnlineImports):
        version = "0.9.0"

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = OldOnlineImports()
    runtime._llm = FakeOnlineLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "hidden_states": [
                            {"layers": "middle", "positions": "last", "capture_mode": "online"}
                        ]
                    },
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "online_hidden_states_unsupported_vllm_version"
    assert exc.value.details == {"supported": "0.10.2", "installed": "0.9.0"}
    assert runtime._llm.generate_calls == 0


def test_generate_extract_rejects_online_hidden_states_without_model_access_surface(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", enable_online_hidden_states=True))
    runtime._imports = FakeOnlineImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {
                    "model": "fake",
                    "prompt": "hello",
                    "extract": {
                        "hidden_states": [
                            {"layers": "middle", "positions": "last", "capture_mode": "online"}
                        ]
                    },
                }
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "online_hidden_states_unavailable"
    assert "apply_model" in exc.value.details["exception"]
    assert runtime._llm.generate_calls == 0


def test_generate_extract_rejects_hidden_states_with_high_gpu_memory_utilization(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.9))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    try:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {
                    "hidden_states": [{"layers": "middle", "positions": "last"}]}}
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    except UnsupportedExtractionError as exc:
        assert exc.code == "hidden_states_unavailable"
        assert exc.details["capability"]["details"]["gpu_memory_utilization"] == 0.9
        assert runtime._pooling_llm is None
        assert runtime._llm.generate_calls == 0
    else:
        raise AssertionError("high GPU memory utilization should be rejected for hidden-state extraction")


def test_replay_hidden_states_still_require_replay_capacity_when_online_is_enabled(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.9, enable_online_hidden_states=True))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate", "pooling"]

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {
                    "hidden_states": [{"layers": "middle", "positions": "last"}]}}
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "hidden_states_unavailable"
    assert exc.value.details["capability"]["details"]["online_hidden_states_enabled"] is False
    assert exc.value.details["capability"]["details"]["gpu_memory_utilization"] == 0.9
    assert runtime._pooling_llm is None
    assert runtime._llm.generate_calls == 0


def test_replay_hidden_states_still_require_pooling_runner_when_online_is_enabled(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits

    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4, enable_online_hidden_states=True))
    runtime._imports = FakePoolingImports()
    runtime._llm = FakeLLM()
    runtime._topology = RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32)
    runtime._supported_runner_types = ["generate"]

    with pytest.raises(UnsupportedExtractionError) as exc:
        runtime.generate_extract(
            ExtractRequest.model_validate(
                {"model": "fake", "prompt": "hello", "extract": {
                    "hidden_states": [{"layers": "middle", "positions": "last"}]}}
            ),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "hidden_states_unavailable"
    assert exc.value.details["capability"]["details"]["supported_runner_types"] == ["generate"]
    assert exc.value.details["capability"]["details"]["online_hidden_states_enabled"] is False
    assert runtime._pooling_llm is None
    assert runtime._llm.generate_calls == 0


def test_runtime_capabilities_mark_hidden_states_unsupported_without_pooling_runner(monkeypatch) -> None:
    class Config:
        supported_runner_types = ["generate"]

    class Engine:
        model_config = Config()

    class LoadableLLM:
        llm_engine = Engine()

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.9.0"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", gpu_memory_utilization=0.4))

    runtime._ensure_loaded()
    capabilities = runtime.capabilities()

    assert capabilities.hidden_states.state == "unsupported"
    assert capabilities.hidden_states.details["supported_runner_types"] == ["generate"]


def test_runtime_capabilities_include_loaded_attention_backend(monkeypatch) -> None:
    class Config:
        attention_backend = "FLASH_ATTN"

    class Engine:
        model_config = Config()

    class LoadableLLM:
        llm_engine = Engine()

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.9.0"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_USE_FLASHINFER", raising=False)
    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake"))

    runtime._ensure_loaded()
    capabilities = runtime.capabilities()

    assert capabilities.attention_backend == "FLASH_ATTN"
    assert capabilities.attentions.details["attention_backend"] == "FLASH_ATTN"


def test_local_files_only_forces_offline_environment(monkeypatch) -> None:
    class LoadableLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Imports:
        module = object()
        version = "0.9.0"
        LLM = LoadableLLM
        SamplingParams = FakeSamplingParams

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    monkeypatch.setattr(vllm_runtime_module, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="fake", local_files_only=True))

    runtime._ensure_loaded()

    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    assert __import__("os").environ["HF_DATASETS_OFFLINE"] == "1"
