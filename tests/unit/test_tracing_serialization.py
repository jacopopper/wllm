from __future__ import annotations

import pytest

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from research.token_baselines import TokenBaselineAdapter
from runtime.capabilities import default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator
from schemas.extraction import ExtractRequest
from tests.unit.test_orchestration import make_inputs
from tracing.serialization import TraceLoadError, load_trace_bundle


def make_persisted_trace(tmp_path):
    request = ExtractRequest.model_validate(
        {
            "model": "fake",
            "prompt": "hello",
            "extract": {"tokens": True, "logprobs": {"top_k": 1}},
        }
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    return orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=True,
    )


def test_load_trace_bundle_validates_and_returns_typed_trace(tmp_path) -> None:
    trace = make_persisted_trace(tmp_path)

    loaded = load_trace_bundle(tmp_path, trace.trace_manifest)
    result = TokenBaselineAdapter().run(loaded)

    assert loaded.id == trace.id
    assert loaded.trace_manifest == trace.trace_manifest
    assert loaded.trace.tokens.token_ids == [10, 11, 12, 13]
    assert result.status == "ok"
    assert result.values["token_count"] == 4


def test_load_trace_bundle_accepts_manifest_dict(tmp_path) -> None:
    trace = make_persisted_trace(tmp_path)

    loaded = load_trace_bundle(tmp_path, trace.trace_manifest.model_dump(mode="json"))

    assert loaded.id == trace.id


def test_load_trace_bundle_rejects_digest_mismatch(tmp_path) -> None:
    trace = make_persisted_trace(tmp_path)
    path = tmp_path / trace.trace_manifest.path
    data = path.read_bytes()
    replacement = b"0" if data[-1:] != b"0" else b"1"
    path.write_bytes(data[:-1] + replacement)

    with pytest.raises(TraceLoadError, match="digest mismatch"):
        load_trace_bundle(tmp_path, trace.trace_manifest)


def test_load_trace_bundle_rejects_path_traversal(tmp_path) -> None:
    trace = make_persisted_trace(tmp_path)
    manifest = trace.trace_manifest.model_copy(update={"path": "../escape.json"})

    with pytest.raises(TraceLoadError, match="escapes"):
        load_trace_bundle(tmp_path, manifest)
