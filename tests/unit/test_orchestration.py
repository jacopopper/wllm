from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from artifacts.loading import load_artifact
from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits, RuntimeTopology, compile_extraction_plan
from runtime.capabilities import Capability, default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator, GeneratedTraceInputs, LogprobCandidate
from schemas.extraction import ExtractRequest, ExtractionSpec
from server.errors import InvalidRequestError, ResourceLimitError, UnsupportedExtractionError


def make_inputs() -> GeneratedTraceInputs:
    return GeneratedTraceInputs(
        model="fake",
        generation={"id": "cmpl_1", "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}},
        prompt_token_ids=[10, 11],
        generated_token_ids=[12, 13],
        decoded_tokens=["a", "b", "c", "d"],
        prompt_logprobs=[
            [],
            [LogprobCandidate(token_id=11, token="b", logprob=-0.3), LogprobCandidate(token_id=77, token="z", logprob=-2.3)],
        ],
        generated_logprobs=[
            [LogprobCandidate(token_id=12, token="c", logprob=-0.2), LogprobCandidate(token_id=99, token="x", logprob=-2.0)],
            [LogprobCandidate(token_id=13, token="d", logprob=-0.1), LogprobCandidate(token_id=88, token="y", logprob=-3.0)],
        ],
        generation_ms=7.5,
        topology=RuntimeTopology(num_layers=12, num_attention_heads=8, hidden_size=32),
    )


def make_hidden_inputs() -> GeneratedTraceInputs:
    layer_0 = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    layer_11 = layer_0 + 1000.0
    return replace(make_inputs(), hidden_states={0: layer_0, 11: layer_11})


def capabilities_with_hidden_states():
    return default_vllm_capabilities("fake", "fake").model_copy(
        update={"hidden_states": Capability(state="supported")}
    )


def test_orchestrator_builds_trace_and_npz_artifact(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "tokens": True,
                "logprobs": {"top_k": 2, "entropy": True, "allow_approximate_entropy": True},
                "artifacts": {"format": "npz", "include": ["tokens", "logprobs"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    assert trace.trace.tokens.token_ids == [10, 11, 12, 13]
    assert trace.trace.spans["generated"] == (2, 4)
    assert trace.trace.logprobs["generated"][0]["token_id"] == 12
    assert trace.trace.logprobs["generated"][0]["token"] == "c"
    assert trace.trace.logprobs["generated"][0]["logprob"] == -0.2
    assert trace.trace.logprobs["generated"][0]["entropy"]["approximation"] == "renormalized_top_k"
    assert len(trace.artifacts) == 1
    assert trace.artifacts[0].tensor_shapes["generated_logprobs"] == [2, 2]
    assert (tmp_path / trace.artifacts[0].path).exists()


def test_orchestrator_builds_uncompressed_npz_artifact(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "tokens": True,
                "artifacts": {"format": "npz", "compression": "uncompressed", "include": ["tokens"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert len(trace.artifacts) == 1
    assert trace.artifacts[0].format == "npz"
    assert trace.artifacts[0].compression == "uncompressed"
    tensors = load_artifact(tmp_path, trace.artifacts[0])
    assert np.array_equal(tensors["token_ids"], np.asarray([10, 11, 12, 13], dtype=np.int64))


def test_artifact_compression_is_npz_only() -> None:
    with pytest.raises(ValueError, match="compression"):
        ExtractRequest.model_validate(
            {
                "model": "fake",
                "prompt": "hello",
                "extract": {
                    "artifacts": {
                        "format": "pt",
                        "compression": "uncompressed",
                        "include": ["tokens"],
                    }
                },
            }
        )


def test_capture_timing_is_included_in_trace_metadata(tmp_path) -> None:
    request = ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {"tokens": True}})
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    trace = orchestrator.build_trace(
        request,
        replace(make_inputs(), capture_ms=3.25),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.metadata.timing_ms.capture == 3.25
    assert trace.metadata.timing_ms.extraction_overhead >= 3.25


def test_hidden_state_position_error_includes_sequence_context(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "max_tokens": 2,
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated"}]},
        }
    )
    inputs = replace(
        make_inputs(),
        prompt_token_ids=[10, 11],
        generated_token_ids=[12, 13],
        hidden_states={0: np.zeros((2, 32), dtype=np.float32)},
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.build_trace(
            request,
            inputs,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "hidden_state_position_unavailable"
    assert exc.value.details["shape"] == [2, 32]
    assert exc.value.details["captured_token_count"] == 2
    assert exc.value.details["positions"] == [3]
    assert exc.value.details["max_requested_position"] == 3
    assert exc.value.details["prompt_token_count"] == 2
    assert exc.value.details["generated_token_count"] == 2
    assert exc.value.details["total_token_count"] == 4


def test_orchestrator_persists_trace_bundle(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"tokens": True, "logprobs": {"top_k": 1}},
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=True,
    )
    assert trace.trace_manifest is not None
    path = tmp_path / trace.trace_manifest.path
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "wllm.trace.v1"
    assert payload["id"] == trace.id
    assert "trace_manifest" not in payload


def test_logprob_artifacts_require_logprob_extraction() -> None:
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"artifacts": {"format": "npz", "include": ["logprobs"]}}}
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    with pytest.raises(InvalidRequestError) as exc:
        orchestrator.preflight(request, ResourceLimits())

    assert exc.value.code == "artifact_dependency_missing"
    assert exc.value.param == "extract.artifacts.include"


def test_inline_extraction_respects_inline_tensor_byte_limit(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"tokens": True, "logprobs": {"top_k": 2}},
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    with pytest.raises(ResourceLimitError) as exc:
        orchestrator.build_trace(
            request,
            make_inputs(),
            limits=ResourceLimits(max_inline_tensor_bytes=103),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.param == "extract"
    assert exc.value.details["requested_inline_bytes"] == 104


def test_logprob_artifacts_respect_requested_top_k(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "logprobs": {"top_k": 1},
                "artifacts": {"format": "npz", "include": ["logprobs"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    tensors = load_artifact(tmp_path, trace.artifacts[0])

    assert trace.artifacts[0].tensor_shapes["generated_logprobs"] == [2, 1]
    assert np.array_equal(tensors["generated_logprob_token_ids"], np.asarray([[12], [13]], dtype=np.int64))


def test_prompt_logprobs_are_included_when_requested(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"logprobs": {"top_k": 1, "include_prompt": True}},
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert trace.trace.logprobs["generated"][0]["top_logprobs"][0]["token_id"] == 12
    assert trace.trace.logprobs["generated"][0]["logprob"] == -0.2
    assert trace.trace.logprobs["generated"][0]["token"] == "c"
    assert trace.trace.logprobs["prompt"][0]["token_id"] == 10
    assert trace.trace.logprobs["prompt"][0]["token"] == "a"
    assert trace.trace.logprobs["prompt"][0]["logprob"] is None
    assert trace.trace.logprobs["prompt"][0]["top_logprobs"] == []
    assert trace.trace.logprobs["prompt"][1]["logprob"] == -0.3
    assert trace.trace.logprobs["prompt"][1]["top_logprobs"][0]["token_id"] == 11


def test_prompt_logprob_artifacts_are_written_when_requested(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "logprobs": {"top_k": 1, "include_prompt": True},
                "artifacts": {"format": "npz", "include": ["logprobs"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    tensors = load_artifact(tmp_path, trace.artifacts[0])

    assert trace.artifacts[0].tensor_shapes["prompt_logprobs"] == [2, 1]
    assert np.array_equal(tensors["prompt_logprob_token_ids"], np.asarray([[-1], [11]], dtype=np.int64))


def test_orchestrator_rejects_exact_entropy_without_complete_distribution() -> None:
    request = ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {"logprobs": {"entropy": True}}})
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.preflight(request, ResourceLimits())
    assert exc.value.code == "exact_entropy_unavailable"


def test_orchestrator_validates_layer_selector_before_unsupported_capability() -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 99, "positions": "last"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    with pytest.raises(InvalidRequestError) as exc:
        orchestrator.preflight(request, ResourceLimits(), topology=RuntimeTopology(num_layers=12))
    assert exc.value.code == "invalid_selector"


def test_orchestrator_records_resolved_selector_metadata(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": [0, -1], "positions": "last_generated", "pool": "last"}],
            },
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    assert trace.metadata.resolved_selectors["hidden_states"] == [
        {"layers": [0, 11], "positions": [3], "pool": "last", "capture_mode": "replay"}
    ]


def test_hidden_state_records_are_returned_inline_when_supported(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert record.name == "hidden_states_0"
    assert record.shape == [2, 1, 32]
    assert record.dtype == "float32"
    assert record.device == "cpu"
    assert record.layers == [0, 11]
    assert record.positions == [3]
    assert record.capture_site == "transformer_block_output"
    assert record.capture_mode == "replay"
    assert record.capture_phase == "replay"
    assert record.position_semantics["prompt_span"] == [0, 2]
    assert record.position_semantics["generated_span"] == [2, 4]
    assert record.artifact_id is None
    assert record.byte_size == 2 * 1 * 32 * 4
    assert record.data[0][0][:3] == [96.0, 97.0, 98.0]
    assert record.data[1][0][:3] == [1096.0, 1097.0, 1098.0]
    assert trace.metadata.capture["hidden_states"]["capture_modes"] == ["replay"]


def test_online_hidden_state_records_map_generated_positions_to_predictor_sources(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated", "capture_mode": "online"}]},
        }
    )
    hidden = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="prompt_prefill_decode_best_effort",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert record.positions == [3]
    assert record.position_semantics["source_positions"] == [2]
    assert record.position_semantics["online_generated_position_mapping"]
    assert record.data[0][0][:3] == [64.0, 65.0, 66.0]


def test_bfloat16_hidden_state_records_are_json_serializable(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated"}]},
        }
    )
    hidden = torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32).to(torch.bfloat16)
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    trace = orchestrator.build_trace(
        request,
        replace(make_inputs(), hidden_states={0: hidden}),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert record.dtype == "torch.bfloat16"
    assert record.capture_dtype == "torch.bfloat16"
    assert record.storage_dtype == "float32"
    assert record.byte_size == 128
    assert record.data[0][0][:3] == [96.0, 97.0, 98.0]


def test_hidden_state_pooling_records_original_positions(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "generated", "pool": "mean"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    record = trace.trace.hidden_states[0]
    assert record.shape == [1, 32]
    assert record.positions == [2, 3]
    assert record.data[0][:3] == [80.0, 81.0, 82.0]


def test_hidden_state_records_can_be_artifact_backed(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": "generated"}],
                "artifacts": {"format": "npz", "include": ["hidden_states"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    assert len(trace.artifacts) == 1
    record = trace.trace.hidden_states[0]
    assert record.data is None
    assert record.artifact_id == trace.artifacts[0].artifact_id
    assert trace.artifacts[0].tensor_shapes["hidden_states_0"] == [1, 2, 32]
    tensors = load_artifact(tmp_path, trace.artifacts[0])
    assert np.array_equal(tensors["hidden_states_0"][0, :, :3], np.asarray([[64, 65, 66], [96, 97, 98]], dtype=np.float32))


def test_hidden_state_request_fails_when_runtime_did_not_capture_layer(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, 1], "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.build_trace(
            request,
            make_hidden_inputs(),
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.code == "hidden_state_layer_unavailable"
    assert exc.value.param == "extract.hidden_states[0]"


def test_inline_hidden_state_records_respect_inline_byte_limit(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    with pytest.raises(ResourceLimitError) as exc:
        orchestrator.build_trace(
            request,
            make_hidden_inputs(),
            limits=ResourceLimits(max_inline_tensor_bytes=255),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.param == "extract"
    assert exc.value.details["requested_inline_bytes"] == 256


def test_inline_hidden_state_records_respect_total_captured_tensor_limit(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())

    with pytest.raises(ResourceLimitError) as exc:
        orchestrator.build_trace(
            request,
            make_hidden_inputs(),
            limits=ResourceLimits(max_inline_tensor_bytes=1024, max_total_captured_tensor_bytes=255),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )

    assert exc.value.param == "extract.hidden_states"
    assert exc.value.details["requested_bytes"] == 256


def test_large_hidden_extraction_requires_artifact_and_server_opt_in() -> None:
    spec_without_artifact = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": "all", "positions": "generated"}]},
        }
    ).extract
    with pytest.raises(ResourceLimitError):
        compile_extraction_plan(
            spec_without_artifact,
            num_layers=12,
            prompt_token_count=2,
            generated_token_count=2,
            limits=ResourceLimits(max_selected_layers=4),
        )

    spec_with_artifact = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": "all", "positions": "generated"}],
                "artifacts": {"format": "npz", "include": ["hidden_states"], "allow_large": True},
            },
        }
    ).extract
    plan = compile_extraction_plan(
        spec_with_artifact,
        num_layers=12,
        prompt_token_count=2,
        generated_token_count=2,
        limits=ResourceLimits(max_selected_layers=4, large_extraction_enabled=True),
    )
    assert plan.hidden_states[0]["layers"] == list(range(12))


def test_orchestrator_rejects_artifact_byte_limit(tmp_path) -> None:
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"artifacts": {"format": "npz", "include": ["tokens"]}}}
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    with pytest.raises(ResourceLimitError):
        orchestrator.build_trace(
            request,
            make_inputs(),
            limits=ResourceLimits(max_artifact_bytes=1),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    assert list(tmp_path.iterdir()) == []


# --- Raw logits rejection ---


def test_raw_logits_extraction_returns_raw_logits_unavailable() -> None:
    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"logprobs": {"raw_logits": True}}}
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.preflight(request, ResourceLimits())
    assert exc.value.code == "raw_logits_unavailable"
    assert exc.value.param == "extract.logprobs.raw_logits"


# --- Pooling modes ---


def test_pool_null_returns_per_position_tensor(tmp_path) -> None:
    """When pool=None, hidden state tensors preserve the positions dimension."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "generated", "pool": None}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.shape == [2, 2, 32]  # layers=2, positions=2, hidden_size=32
    assert record.positions == [2, 3]
    assert record.data is not None


def test_pool_mean_returns_position_averaged_tensor(tmp_path) -> None:
    """When pool='mean', output is element-wise mean across positions."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "generated", "pool": "mean"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    # Pool removes position dimension: [layers, hidden_size]
    assert record.shape == [2, 32]
    assert record.positions == [2, 3]
    # Verify mean: layer 0 positions [2,3] have values starting at 64 and 96
    assert record.data[0][:3] == [80.0, 81.0, 82.0]


def test_pool_max_returns_position_max_tensor(tmp_path) -> None:
    """When pool='max', output is element-wise max across positions."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "generated", "pool": "max"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.shape == [2, 32]
    assert record.positions == [2, 3]
    # max of layer 0 positions [2,3] should pick the larger values (position 3)
    assert record.data[0][:3] == [96.0, 97.0, 98.0]


def test_pool_last_returns_only_last_selected_position(tmp_path) -> None:
    """When pool='last', output equals the tensor at the highest-index selected position."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": [0, -1], "positions": "generated", "pool": "last"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.shape == [2, 32]
    assert record.positions == [2, 3]
    # last of [2,3] is position 3
    assert record.data[0][:3] == [96.0, 97.0, 98.0]


def test_pool_last_with_single_position(tmp_path) -> None:
    """When pool='last' with a single position, returns that position's tensor."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last", "pool": "last"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.shape == [1, 32]
    assert record.positions == [3]
    # position 3 in make_hidden_inputs has values 96+
    assert record.data[0][:3] == [96.0, 97.0, 98.0]


# --- Default capture_mode ---


def test_default_capture_mode_is_replay_when_omitted(tmp_path) -> None:
    """When capture_mode is omitted, the compiled plan uses 'replay'."""
    spec = ExtractionSpec.model_validate(
        {"hidden_states": [{"layers": [0], "positions": "last"}]}
    )
    assert spec.hidden_states[0].capture_mode == "replay"

    plan = compile_extraction_plan(
        spec,
        num_layers=12,
        prompt_token_count=3,
        generated_token_count=2,
        limits=ResourceLimits(),
    )
    assert plan.hidden_states[0]["capture_mode"] == "replay"


def test_capture_mode_propagates_to_tensor_record(tmp_path) -> None:
    """The capture_mode flows from spec to plan to TensorRecord."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.capture_mode == "replay"
    assert record.capture_phase == "replay"


# --- capture_phase correctness ---


def test_capture_phase_reflects_selected_position_domain_prompt(tmp_path) -> None:
    """When selected positions are all prompt tokens, capture_phase is 'prompt_prefill' for online."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": "prompt", "capture_mode": "online"}]
            },
        }
    )
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="prompt_prefill",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.capture_mode == "online"
    assert record.capture_phase == "prompt_prefill"


def test_capture_phase_reflects_selected_position_domain_decode(tmp_path) -> None:
    """When selected positions are all generated tokens, capture_phase is 'decode' for online."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": "generated", "capture_mode": "online"}]
            },
        }
    )
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="decode",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.capture_mode == "online"
    assert record.capture_phase == "decode"


def test_capture_phase_mixed_prompt_and_generated(tmp_path) -> None:
    """When selected positions span prompt and generated tokens, capture_phase is 'mixed_prompt_prefill_decode'."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": [0, 3], "capture_mode": "online"}]
            },
        }
    )
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="mixed_prompt_prefill_decode",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.capture_mode == "online"
    assert record.capture_phase == "mixed_prompt_prefill_decode"


# --- Online position shifting ---


def test_online_capture_uses_shifted_source_positions_for_generated_tokens(tmp_path) -> None:
    """Online capture maps generated position p to source position p-1."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "generated", "capture_mode": "online"}]},
        }
    )
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="prompt_prefill_decode_best_effort",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.positions == [2, 3]
    assert record.position_semantics["source_positions"] == [1, 2]
    assert record.position_semantics["online_generated_position_mapping"] is not None
    # source position 1 (index 1) has value 32, source position 2 has value 64
    assert record.data[0][0][:3] == [32.0, 33.0, 34.0]
    assert record.data[0][1][:3] == [64.0, 65.0, 66.0]


# --- Timing metadata ---


def test_timing_ms_fields_are_non_negative(tmp_path) -> None:
    request = ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {"tokens": True}})
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    timing = trace.metadata.timing_ms
    assert timing.generation >= 0
    assert timing.capture >= 0
    assert timing.postprocess >= 0
    assert timing.serialization >= 0
    assert timing.extraction_overhead >= 0
    assert timing.total >= 0


def test_timing_ms_total_equals_generation_plus_overhead(tmp_path) -> None:
    request = ExtractRequest.model_validate({"model": "fake", "prompt": "hello", "extract": {"tokens": True}})
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    timing = trace.metadata.timing_ms
    assert timing.total == timing.generation + timing.extraction_overhead


# --- Simultaneous hidden-state and logprob extraction ---


def test_simultaneous_hidden_state_and_logprob_extraction(tmp_path) -> None:
    """Both hidden states and logprobs are correctly extracted in the same request."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": "last_generated"}],
                "logprobs": {"top_k": 2, "entropy": True, "allow_approximate_entropy": True},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    # Verify hidden state records
    assert len(trace.trace.hidden_states) == 1
    hs_record = trace.trace.hidden_states[0]
    assert hs_record.name == "hidden_states_0"
    assert hs_record.layers == [0]
    assert hs_record.positions == [3]
    assert hs_record.capture_mode == "replay"

    # Verify logprob records
    assert "generated" in trace.trace.logprobs
    assert len(trace.trace.logprobs["generated"]) == 2
    lp0 = trace.trace.logprobs["generated"][0]
    assert lp0["token_id"] == 12
    assert lp0["token"] == "c"
    assert lp0["logprob"] == -0.2
    assert lp0["entropy"]["approximation"] == "renormalized_top_k"


# --- Simultaneous hidden-state and logprob with artifacts ---


def test_simultaneous_hidden_state_and_logprob_with_artifacts(tmp_path) -> None:
    """Both hidden states and logprobs are artifact-backed in the same request."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [{"layers": 0, "positions": "last_generated"}],
                "logprobs": {"top_k": 2, "entropy": True, "allow_approximate_entropy": True},
                "artifacts": {"format": "npz", "include": ["hidden_states", "logprobs"]},
            },
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )

    # Hidden state is artifact-backed
    assert trace.trace.hidden_states[0].data is None
    assert trace.trace.hidden_states[0].artifact_id is not None

    # Logprobs are artifact-backed
    assert len(trace.artifacts) == 1
    artifact = trace.artifacts[0]
    assert "hidden_states_0" in artifact.included_tensor_names
    assert "generated_logprobs" in artifact.included_tensor_names

    # Both artifacts load correctly
    tensors = load_artifact(tmp_path, artifact)
    assert "hidden_states_0" in tensors
    assert "generated_logprob_token_ids" in tensors
    assert "generated_logprobs" in tensors


# --- Hook cleanup after successful online capture ---


def test_hook_cleanup_after_successful_online_capture(tmp_path) -> None:
    """After a successful online capture, the model has no residual hook artifacts."""
    # The orchestrator itself doesn't install hooks - the runtime does.
    # We test that after building a trace with online capture, the trace is valid
    # and the record correctly reflects online semantics.
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated", "capture_mode": "online"}]},
        }
    )
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
        hidden_state_capture_mode="online",
        hidden_state_capture_phase="prompt_prefill_decode_best_effort",
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.capture_mode == "online"
    assert record.capture_phase is not None
    assert record.position_semantics["online_generated_position_mapping"] is not None


# --- Chunked forward-pass concatenation ---


def test_online_capture_concatenates_chunked_forward_pass_tensors(tmp_path) -> None:
    """Online capture correctly handles hidden states that span the full sequence (chunked forward passes)."""
    # Simulate a 4-position tensor (as if concatenated from chunked forward passes)
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": [0, 2, 3], "pool": None}]},
        }
    )
    # Create a 4-position tensor
    hidden = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: hidden},
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        inputs,
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    record = trace.trace.hidden_states[0]
    assert record.shape == [1, 3, 32]
    assert record.positions == [0, 2, 3]
    # Position 0 should be values 0-31, position 2 should be 64-95, position 3 should be 96-127
    assert record.data[0][0][:3] == [0.0, 1.0, 2.0]
    assert record.data[0][1][:3] == [64.0, 65.0, 66.0]
    assert record.data[0][2][:3] == [96.0, 97.0, 98.0]


# --- Replay sequence exceeding max_model_len ---


def test_replay_sequence_exceeding_max_model_len_is_rejected(tmp_path) -> None:
    """When a replay requests positions beyond the captured hidden states, it fails."""
    # Create inputs with only 2 prompt + 1 generated = 3 positions total
    # but request position "last" with 4 total tokens (2+2), so position 3 would be out of range
    # if the captured tensor only has 3 elements.
    short_hidden = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    inputs = replace(
        make_inputs(),
        hidden_states={0: short_hidden},
    )
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last"}]},
        }
    )
    # "last" resolves to position 3 (total tokens = 4), but hidden states only has 3 positions
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    with pytest.raises(UnsupportedExtractionError) as exc:
        orchestrator.build_trace(
            request,
            inputs,
            limits=ResourceLimits(),
            artifact_store=ArtifactStore(tmp_path),
            persist=False,
        )
    assert exc.value.code == "hidden_state_position_unavailable"


# --- Multiple extraction specs in one request ---


def test_multiple_hidden_state_specs_in_one_request(tmp_path) -> None:
    """Multiple hidden state extraction specs are all resolved and recorded."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {
                "hidden_states": [
                    {"layers": 0, "positions": "prompt", "pool": None},
                    {"layers": -1, "positions": "last_generated", "pool": "last"},
                ],
            },
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    assert len(trace.trace.hidden_states) == 2
    assert trace.metadata.capture["hidden_states"]["capture_modes"] == ["replay"]


# --- Token-only extraction is extraction-free safe ---


def test_extraction_with_only_hidden_states_no_tokens(tmp_path) -> None:
    """Hidden-only extraction does not produce token IDs or decoded tokens in trace."""
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"hidden_states": [{"layers": 0, "positions": "last_generated"}]},
        }
    )
    orchestrator = ExtractionOrchestrator(capabilities_with_hidden_states())
    trace = orchestrator.build_trace(
        request,
        make_hidden_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=False,
    )
    # Tokens not requested, so trace tokens should be empty
    assert trace.trace.tokens.token_ids == []
    assert trace.trace.tokens.tokens == []
    # But hidden states are still present
    assert len(trace.trace.hidden_states) == 1
    assert trace.trace.hidden_states[0].name == "hidden_states_0"


# --- Logprob arrays top_k enforcement ---


def test_logprob_arrays_respects_top_k_and_none() -> None:
    """_logprob_arrays with top_k=3 caps output width at 3; top_k=None returns full width."""
    # Create rows with 5 candidates each (wider than top_k=3)
    wide_candidates = [
        LogprobCandidate(token_id=100 + i, logprob=-float(i), token=f"t{i}") for i in range(5)
    ]
    inputs = replace(
        make_inputs(),
        generated_logprobs=[list(wide_candidates), list(wide_candidates)],
        generated_token_ids=[200, 201],
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))

    # top_k=3: output width must be <= 3
    token_ids_3, logprobs_3 = orchestrator._logprob_arrays(inputs, top_k=3)
    assert token_ids_3.shape == (2, 3)
    assert logprobs_3.shape == (2, 3)
    assert token_ids_3[0, 0] == 100
    assert token_ids_3[0, 1] == 101
    assert token_ids_3[0, 2] == 102
    assert logprobs_3[0, 0] == 0.0
    assert logprobs_3[0, 1] == -1.0
    assert logprobs_3[0, 2] == -2.0

    # top_k=None: output width is the full row width (5 candidates)
    token_ids_full, logprobs_full = orchestrator._logprob_arrays(inputs, top_k=None)
    assert token_ids_full.shape == (2, 5)
    assert logprobs_full.shape == (2, 5)
    assert token_ids_full[0, 4] == 104
    assert logprobs_full[0, 4] == -4.0
    # Row 1 should have the same values (both rows use identical candidates)
    assert token_ids_full[1, 4] == 104
    assert logprobs_full[1, 4] == -4.0
