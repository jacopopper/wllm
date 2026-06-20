from __future__ import annotations

import importlib.util
import os

import pytest


pytestmark = pytest.mark.integration

HAS_VLLM = importlib.util.find_spec("vllm") is not None
REQUIRES_LOCAL_VLLM_MODEL = pytest.mark.skipif(
    not os.environ.get("WLLM_TEST_MODEL") or not HAS_VLLM,
    reason="Set WLLM_TEST_MODEL to a local model and install vLLM to run integration tests.",
)


def _runtime_config(model: str, *, enable_online_hidden_states: bool = False):
    from runtime.vllm_runtime import VLLMRuntimeConfig

    return VLLMRuntimeConfig(
        model=model,
        local_files_only=True,
        max_model_len=int(os.environ.get("WLLM_TEST_MAX_MODEL_LEN", "1024")),
        gpu_memory_utilization=float(os.environ.get("WLLM_TEST_GPU_MEMORY_UTILIZATION", "0.35")),
        enable_online_hidden_states=enable_online_hidden_states,
    )


@REQUIRES_LOCAL_VLLM_MODEL
def test_vllm_token_logprob_extraction_smoke(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.vllm_runtime import VLLMRuntime
    from schemas.extraction import ExtractRequest

    model = os.environ["WLLM_TEST_MODEL"]
    runtime = VLLMRuntime(_runtime_config(model))
    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {"model": model, "prompt": "Hello", "max_tokens": 2, "extract": {"tokens": True, "logprobs": {"top_k": 1}}}
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    assert trace.schema_version == "wllm.trace.v1"
    assert trace.trace.tokens.token_ids
    assert "generated" in trace.trace.logprobs


@REQUIRES_LOCAL_VLLM_MODEL
def test_vllm_hidden_state_extraction_smoke(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.vllm_runtime import VLLMRuntime
    from schemas.extraction import ExtractRequest

    model = os.environ["WLLM_TEST_MODEL"]
    runtime = VLLMRuntime(_runtime_config(model))
    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {
                "model": model,
                "prompt": "Hello",
                "max_tokens": 2,
                "extract": {
                    "hidden_states": [{"layers": "middle", "positions": "last_generated"}],
                },
            }
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.schema_version == "wllm.trace.v1"
    assert trace.trace.hidden_states
    record = trace.trace.hidden_states[0]
    assert record.shape
    assert record.dtype
    assert record.device
    assert len(record.layers) == 1
    assert record.layers[0] >= 0
    assert record.positions
    assert record.capture_site == "transformer_block_output"
    assert record.data is not None or record.artifact_id is not None


@REQUIRES_LOCAL_VLLM_MODEL
def test_vllm_online_hidden_state_extraction_smoke(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.vllm_runtime import VLLMRuntime
    from schemas.extraction import ExtractRequest

    model = os.environ["WLLM_TEST_MODEL"]
    runtime = VLLMRuntime(_runtime_config(model, enable_online_hidden_states=True))
    trace = runtime.generate_extract(
        ExtractRequest.model_validate(
            {
                "model": model,
                "prompt": "Hello",
                "max_tokens": 2,
                "temperature": 0.0,
                "extract": {
                    "hidden_states": [
                        {"layers": "middle", "positions": "prompt", "capture_mode": "online"}
                    ],
                },
            }
        ),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.schema_version == "wllm.trace.v1"
    assert trace.trace.hidden_states
    record = trace.trace.hidden_states[0]
    prompt_tokens = trace.generation["usage"]["prompt_tokens"]
    assert record.capture_site == "transformer_block_output"
    assert record.capture_mode == "online"
    assert record.capture_phase == "prompt_prefill"
    assert record.capture_metadata["hook_scope"] == "active_generation_runner"
    assert record.capture_metadata["layer_chunk_shapes"]
    assert record.capture_metadata["capture_filter"]["type"] == "dense_prefix"
    assert record.capture_metadata["capture_filter"]["max_source_position"] == prompt_tokens - 1
    assert record.positions
    assert all(0 <= position < prompt_tokens for position in record.positions)
    assert record.shape[0] == len(record.layers)
    assert record.shape[1] == len(record.positions)
    assert record.data is not None or record.artifact_id is not None
    assert trace.metadata.timing_ms.capture > 0


@REQUIRES_LOCAL_VLLM_MODEL
def test_vllm_hidden_state_extraction_replays_full_sequence_repeatedly(tmp_path) -> None:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.vllm_runtime import VLLMRuntime
    from schemas.extraction import ExtractRequest

    model = os.environ["WLLM_TEST_MODEL"]
    runtime = VLLMRuntime(_runtime_config(model))
    runtime.prewarm_hidden_states()
    assert runtime._pooling_llm is not None
    prompt = (
        "Summarize this implementation note in one concise sentence. "
        "Hidden-state extraction must replay the complete final token sequence "
        "on every request so transformer hooks see prompt and generated tokens."
    )
    request = ExtractRequest.model_validate(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0.0,
            "extract": {
                "hidden_states": [{"layers": "middle", "positions": "last_generated"}],
            },
        }
    )

    traces = [
        runtime.generate_extract(
            request,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
        for _ in range(2)
    ]

    for trace in traces:
        record = trace.trace.hidden_states[0]
        assert record.positions
        assert record.positions[-1] == trace.generation["usage"]["total_tokens"] - 1
        assert record.shape[1] == len(record.positions)
        assert trace.metadata.timing_ms.capture > 0
