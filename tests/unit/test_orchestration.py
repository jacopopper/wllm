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
from schemas.extraction import ExtractRequest
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
        {"layers": [0, 11], "positions": [3], "pool": "last"}
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
    assert record.artifact_id is None
    assert record.byte_size == 2 * 1 * 32 * 4
    assert record.data[0][0][:3] == [96.0, 97.0, 98.0]
    assert record.data[1][0][:3] == [1096.0, 1097.0, 1098.0]


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
