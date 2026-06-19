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


def _runtime_config(model: str):
    from runtime.vllm_runtime import VLLMRuntimeConfig

    return VLLMRuntimeConfig(
        model=model,
        local_files_only=True,
        max_model_len=int(os.environ.get("WLLM_TEST_MAX_MODEL_LEN", "1024")),
        gpu_memory_utilization=float(os.environ.get("WLLM_TEST_GPU_MEMORY_UTILIZATION", "0.35")),
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
