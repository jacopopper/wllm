from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest

import artifacts.store as store_module
from artifacts.loading import ArtifactLoadError, load_artifact
from artifacts.store import ArtifactStore
from artifacts.torch import TorchArtifactUnavailableError
from server.errors import InvalidRequestError, UnsupportedExtractionError
from tracing.context import active_trace_id


def test_npz_artifact_manifest(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"hidden": np.zeros((2, 3), dtype=np.float32)}, format="npz")
    assert manifest.format == "npz"
    assert manifest.byte_size > 0
    assert manifest.tensor_shapes == {"hidden": [2, 3]}
    assert manifest.tensor_dtypes == {"hidden": "float32"}
    assert (tmp_path / manifest.path).exists()


def test_load_npz_artifact_validates_manifest(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"hidden": np.ones((2, 3), dtype=np.float32)}, format="npz")

    tensors = load_artifact(tmp_path, manifest)

    assert np.array_equal(tensors["hidden"], np.ones((2, 3), dtype=np.float32))


def test_npz_artifact_can_be_uncompressed(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(
        trace_id="trace_1",
        tensors={"hidden": np.arange(6, dtype=np.float32).reshape(2, 3)},
        format="npz",
        compression="uncompressed",
    )

    assert manifest.format == "npz"
    assert manifest.compression == "uncompressed"
    with zipfile.ZipFile(tmp_path / manifest.path) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
    tensors = load_artifact(tmp_path, manifest)
    assert np.array_equal(tensors["hidden"], np.arange(6, dtype=np.float32).reshape(2, 3))


def test_pt_artifact_rejects_compression_option(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(InvalidRequestError) as exc:
        store.put(
            trace_id="trace_1",
            tensors={"hidden": np.arange(6, dtype=np.float32).reshape(2, 3)},
            format="pt",
            compression="uncompressed",
        )

    assert exc.value.param == "extract.artifacts.compression"


def test_npz_artifact_converts_bfloat16_tensors(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)

    manifest = store.put(trace_id="trace_1", tensors={"hidden": tensor}, format="npz")
    tensors = load_artifact(tmp_path, manifest)

    assert manifest.tensor_capture_dtypes == {"hidden": "torch.bfloat16"}
    assert manifest.tensor_storage_dtypes == {"hidden": "float32"}
    assert manifest.tensor_dtypes == {"hidden": "float32"}
    assert tensors["hidden"].dtype == np.float32
    assert np.array_equal(tensors["hidden"], np.arange(6, dtype=np.float32).reshape(2, 3))


def test_load_artifact_accepts_manifest_dict(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1, 2], dtype=np.int64)}, format="npz")

    tensors = load_artifact(tmp_path, manifest.model_dump(mode="json"))

    assert np.array_equal(tensors["tokens"], np.asarray([1, 2], dtype=np.int64))


def test_npz_artifact_manifest_with_relative_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = ArtifactStore(__import__("pathlib").Path("relative-artifacts"))
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1, 2, 3], dtype=np.int64)}, format="npz")
    assert manifest.path.endswith(".npz")
    assert not manifest.path.startswith("/")
    assert (tmp_path / "relative-artifacts" / manifest.path).exists()


def test_pt_artifact_manifest_uses_tensor_metadata(tmp_path, monkeypatch) -> None:
    def fake_save_pt(path, tensors):
        del tensors
        path.write_bytes(b"pt")

    monkeypatch.setattr(store_module, "save_pt", fake_save_pt)
    store = ArtifactStore(tmp_path)
    manifest = store.put(
        trace_id="trace_1",
        tensors={"scores": np.zeros((2, 3), dtype=np.float32)},
        format="pt",
    )
    assert manifest.format == "pt"
    assert manifest.tensor_shapes == {"scores": [2, 3]}
    assert manifest.tensor_dtypes == {"scores": "float32"}


def test_pt_artifact_with_numpy_arrays_loads_with_weights_only(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path)
    manifest = store.put(
        trace_id="trace_1",
        tensors={"scores": np.arange(6, dtype=np.float32).reshape(2, 3)},
        format="pt",
    )

    tensors = load_artifact(tmp_path, manifest)

    assert manifest.tensor_capture_dtypes == {"scores": "float32"}
    assert manifest.tensor_storage_dtypes == {"scores": "torch.float32"}
    assert manifest.tensor_dtypes == {"scores": "torch.float32"}
    assert torch.is_tensor(tensors["scores"])
    assert tensors["scores"].shape == (2, 3)
    assert tensors["scores"].dtype == torch.float32


def test_pt_artifact_unavailable_is_structured(tmp_path, monkeypatch) -> None:
    def fake_save_pt(path, tensors):
        del path, tensors
        raise TorchArtifactUnavailableError("torch missing")

    monkeypatch.setattr(store_module, "save_pt", fake_save_pt)
    store = ArtifactStore(tmp_path)
    with pytest.raises(UnsupportedExtractionError) as exc:
        store.put(trace_id="trace_1", tensors={"scores": np.zeros((2, 3), dtype=np.float32)}, format="pt")
    assert exc.value.code == "pt_artifacts_unavailable"
    assert exc.value.param == "extract.artifacts.format"


def test_load_artifact_rejects_digest_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1], dtype=np.int64)}, format="npz")
    path = tmp_path / manifest.path
    data = path.read_bytes()
    replacement = b"0" if data[-1:] != b"0" else b"1"
    path.write_bytes(data[:-1] + replacement)

    with pytest.raises(ArtifactLoadError, match="digest mismatch"):
        load_artifact(tmp_path, manifest)


def test_load_artifact_rejects_path_traversal(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1], dtype=np.int64)}, format="npz")
    manifest = manifest.model_copy(update={"path": "../escape.npz"})

    with pytest.raises(ArtifactLoadError, match="escapes"):
        load_artifact(tmp_path, manifest)


def test_npz_write_failure_leaves_no_partial_artifact(tmp_path, monkeypatch) -> None:
    def fail_save(path, tensors, **kwargs):
        del tensors, kwargs
        path.write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(store_module, "save_npz", fail_save)
    store = ArtifactStore(tmp_path)

    with pytest.raises(RuntimeError, match="write failed"):
        store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1], dtype=np.int64)}, format="npz")

    assert list(tmp_path.iterdir()) == []


def test_trace_bundle_manifest(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1"},
    )
    path = tmp_path / manifest.path
    assert manifest.object == "wllm.trace_manifest"
    assert manifest.trace_id == "trace_1"
    assert manifest.byte_size == path.stat().st_size
    assert json.loads(path.read_text())["id"] == "trace_1"


def test_artifact_path_safety(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(InvalidRequestError):
        store._artifact_path("../escape.npz")


def test_artifact_trace_context_mismatch_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    token = active_trace_id.set("trace_active")
    try:
        with pytest.raises(InvalidRequestError) as exc:
            store.put(trace_id="trace_other", tensors={"tokens": np.asarray([1], dtype=np.int64)}, format="npz")
        assert exc.value.code == "trace_context_mismatch"
    finally:
        active_trace_id.reset(token)


def test_artifact_store_does_not_create_root_until_write(tmp_path) -> None:
    unused_root = tmp_path / "unused-artifacts"
    store = ArtifactStore(unused_root)
    assert not unused_root.exists()


def test_artifact_store_creates_root_on_first_put(tmp_path) -> None:
    artifact_root = tmp_path / "created-on-first-put"
    store = ArtifactStore(artifact_root)
    assert not artifact_root.exists()
    store.put(trace_id="trace_1", tensors={"a": np.asarray([1], dtype=np.int64)}, format="npz")
    assert artifact_root.exists()


# ---------------------------------------------------------------------------
# NPZ default compression verification
# ---------------------------------------------------------------------------


def test_npz_default_compression_uses_deflated(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(
        trace_id="trace_1",
        tensors={"hidden": np.arange(24, dtype=np.float32).reshape(2, 3, 4)},
        format="npz",
    )
    assert manifest.compression == "compressed"
    with zipfile.ZipFile(tmp_path / manifest.path) as archive:
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED


# ---------------------------------------------------------------------------
# PT atomic write cleanup on failure
# ---------------------------------------------------------------------------


def test_pt_write_failure_leaves_no_partial_artifact(tmp_path, monkeypatch) -> None:
    def fail_save_pt(path, tensors):
        del tensors
        path.write_bytes(b"partial-pt")
        raise RuntimeError("pt write failed")

    monkeypatch.setattr(store_module, "save_pt", fail_save_pt)
    store = ArtifactStore(tmp_path)

    with pytest.raises(RuntimeError, match="pt write failed"):
        store.put(trace_id="trace_1", tensors={"scores": np.zeros((2, 3), dtype=np.float32)}, format="pt")

    # Neither .tmp files nor the destination path should remain
    remaining = list(tmp_path.iterdir())
    assert remaining == []


# ---------------------------------------------------------------------------
# Multi-tensor NPZ manifest and round-trip
# ---------------------------------------------------------------------------


def test_multi_tensor_npz_manifest_shapes_dtypes(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    tensors = {
        "alpha": np.zeros((2, 3), dtype=np.float32),
        "beta": np.ones((4,), dtype=np.int64),
        "gamma": np.full((3, 5), True, dtype=np.bool_),
    }
    manifest = store.put(trace_id="trace_multi", tensors=tensors, format="npz")

    assert set(manifest.included_tensor_names) == {"alpha", "beta", "gamma"}
    assert manifest.tensor_shapes == {"alpha": [2, 3], "beta": [4], "gamma": [3, 5]}
    assert manifest.tensor_dtypes == {"alpha": "float32", "beta": "int64", "gamma": "bool"}


def test_multi_tensor_npz_round_trip(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    original = {
        "alpha": np.arange(6, dtype=np.float32).reshape(2, 3),
        "beta": np.array([10, 20, 30], dtype=np.int64),
        "gamma": np.array([True, False, True], dtype=np.bool_),
    }
    manifest = store.put(trace_id="trace_multi", tensors=original, format="npz")
    loaded = load_artifact(tmp_path, manifest)

    assert set(loaded) == set(original)
    for name in original:
        assert np.array_equal(loaded[name], original[name]), f"round-trip mismatch for tensor {name!r}"


def test_manifest_capture_storage_dtypes_all_tensors(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    tensors = {
        "a": np.zeros((2,), dtype=np.float32),
        "b": np.ones((3,), dtype=np.int64),
    }
    manifest = store.put(trace_id="trace_1", tensors=tensors, format="npz")

    assert set(manifest.tensor_capture_dtypes) == set(manifest.included_tensor_names)
    assert set(manifest.tensor_storage_dtypes) == set(manifest.included_tensor_names)


# ---------------------------------------------------------------------------
# load_artifact byte_size / shape / dtype mismatch
# ---------------------------------------------------------------------------


def test_load_artifact_rejects_byte_size_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1, 2], dtype=np.int64)}, format="npz")
    manifest = manifest.model_copy(update={"byte_size": 999999})

    with pytest.raises(ArtifactLoadError, match="byte size mismatch"):
        load_artifact(tmp_path, manifest)


def test_load_artifact_rejects_shape_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.zeros((2, 3), dtype=np.float32)}, format="npz")
    manifest = manifest.model_copy(update={"tensor_shapes": {"tokens": [99, 99]}})

    with pytest.raises(ArtifactLoadError, match="shape mismatch"):
        load_artifact(tmp_path, manifest)


def test_load_artifact_rejects_dtype_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.zeros((2, 3), dtype=np.float32)}, format="npz")
    manifest = manifest.model_copy(update={"tensor_storage_dtypes": {"tokens": "float64"}})

    with pytest.raises(ArtifactLoadError, match="dtype mismatch"):
        load_artifact(tmp_path, manifest)


# ---------------------------------------------------------------------------
# load_artifact corrupt / missing file
# ---------------------------------------------------------------------------


def test_load_artifact_rejects_corrupt_npz(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1, 2], dtype=np.int64)}, format="npz")
    # Write corrupt data, then update manifest so byte_size/sha256 checks pass
    corrupt = b"not a zip file"
    (tmp_path / manifest.path).write_bytes(corrupt)
    manifest = manifest.model_copy(update={
        "byte_size": len(corrupt),
        "sha256": hashlib.sha256(corrupt).hexdigest(),
    })

    with pytest.raises(ArtifactLoadError, match="NPZ artifact could not be decoded"):
        load_artifact(tmp_path, manifest)


def test_load_artifact_rejects_corrupt_pt(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path)
    manifest = store.put(
        trace_id="trace_1",
        tensors={"scores": np.arange(6, dtype=np.float32).reshape(2, 3)},
        format="pt",
    )
    # Write corrupt data, then update manifest so byte_size/sha256 checks pass
    corrupt = b"not a valid torch archive"
    (tmp_path / manifest.path).write_bytes(corrupt)
    manifest = manifest.model_copy(update={
        "byte_size": len(corrupt),
        "sha256": hashlib.sha256(corrupt).hexdigest(),
    })

    with pytest.raises(ArtifactLoadError, match="PT artifact could not be decoded"):
        load_artifact(tmp_path, manifest)


def test_load_artifact_rejects_missing_file(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.put(trace_id="trace_1", tensors={"tokens": np.asarray([1], dtype=np.int64)}, format="npz")
    (tmp_path / manifest.path).unlink()

    with pytest.raises(ArtifactLoadError, match="could not be read"):
        load_artifact(tmp_path, manifest)


# ---------------------------------------------------------------------------
# Dtype round-trips (int64, float16, bool) through NPZ
# ---------------------------------------------------------------------------


def test_npz_roundtrip_int64(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    original = {"ids": np.array([-(2**32), -1, 0, 1, 2**32 - 1], dtype=np.int64)}
    manifest = store.put(trace_id="trace_1", tensors=original, format="npz")
    loaded = load_artifact(tmp_path, manifest)
    assert np.array_equal(loaded["ids"], original["ids"])
    assert loaded["ids"].dtype == np.int64


def test_npz_roundtrip_float16(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    original = {"vals": np.array([1.5, 2.5, 3.5], dtype=np.float16)}
    manifest = store.put(trace_id="trace_1", tensors=original, format="npz")
    loaded = load_artifact(tmp_path, manifest)
    assert np.array_equal(loaded["vals"], original["vals"])
    assert loaded["vals"].dtype == np.float16


def test_npz_roundtrip_bool(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    original = {"flags": np.array([True, False, True, True], dtype=np.bool_)}
    manifest = store.put(trace_id="trace_1", tensors=original, format="npz")
    loaded = load_artifact(tmp_path, manifest)
    assert np.array_equal(loaded["flags"], original["flags"])
    assert loaded["flags"].dtype == np.bool_


# ---------------------------------------------------------------------------
# PT round-trip with int64 and float16 dtypes
# ---------------------------------------------------------------------------


def test_pt_roundtrip_int64_float16(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path)
    original = {
        "ids": np.array([100, 200, 300], dtype=np.int64),
        "vals": np.array([1.5, 2.5, 3.5], dtype=np.float16),
    }
    manifest = store.put(trace_id="trace_1", tensors=original, format="pt")
    loaded = load_artifact(tmp_path, manifest)

    assert torch.is_tensor(loaded["ids"])
    assert torch.is_tensor(loaded["vals"])
    assert torch.equal(loaded["ids"], torch.from_numpy(original["ids"]))
    assert torch.equal(loaded["vals"].to(torch.float16), torch.from_numpy(original["vals"]).to(torch.float16))


# ---------------------------------------------------------------------------
# Trace bundle: dict manifest, digest, path traversal, typed trace
# ---------------------------------------------------------------------------


def test_load_trace_bundle_accepts_dict_manifest(tmp_path) -> None:
    from tracing.serialization import load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )
    trace_model = load_trace_bundle(tmp_path, manifest)
    trace_dict = load_trace_bundle(tmp_path, manifest.model_dump(mode="json"))
    assert trace_model.id == trace_dict.id
    assert trace_model.schema_version == trace_dict.schema_version


def test_load_trace_bundle_rejects_digest_mismatch(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )
    manifest = manifest.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(TraceLoadError, match="digest mismatch"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_rejects_path_traversal(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )
    manifest = manifest.model_copy(update={"path": "../escape.json"})

    with pytest.raises(TraceLoadError, match="escapes"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_returns_typed_trace(tmp_path) -> None:
    from schemas.traces import TraceEnvelope
    from tracing.serialization import load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )
    trace = load_trace_bundle(tmp_path, manifest)
    assert isinstance(trace, TraceEnvelope)
    assert trace.schema_version == manifest.schema_version
    assert trace.trace_manifest == manifest


# ---------------------------------------------------------------------------
# Trace bundle: trace_id / schema_version mismatch, corrupt JSON/UTF-8
# ---------------------------------------------------------------------------


def test_load_trace_bundle_rejects_trace_id_mismatch(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "wrong_trace_id", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )

    with pytest.raises(TraceLoadError, match="Trace ID mismatch"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_rejects_schema_version_mismatch(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                  "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}},
    )
    # Tamper the manifest's schema_version so it differs from the payload
    manifest = manifest.model_copy(update={"schema_version": "wllm.trace.v99"})

    with pytest.raises(TraceLoadError, match="schema version mismatch"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_rejects_invalid_utf8(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {}, "trace": {}},
    )
    # Write invalid UTF-8 bytes and update manifest
    bad_bytes = b'\x80\x81\x82'
    (tmp_path / manifest.path).write_bytes(bad_bytes)
    manifest = manifest.model_copy(update={
        "byte_size": len(bad_bytes),
        "sha256": hashlib.sha256(bad_bytes).hexdigest(),
    })

    with pytest.raises(TraceLoadError, match="valid UTF-8"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_rejects_invalid_json(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {}, "trace": {}},
    )
    # Write non-JSON content and update manifest
    bad_text = "this is not json"
    bad_bytes = bad_text.encode("utf-8")
    (tmp_path / manifest.path).write_bytes(bad_bytes)
    manifest = manifest.model_copy(update={
        "byte_size": len(bad_bytes),
        "sha256": hashlib.sha256(bad_bytes).hexdigest(),
    })

    with pytest.raises(TraceLoadError, match="valid UTF-8 JSON"):
        load_trace_bundle(tmp_path, manifest)


def test_load_trace_bundle_rejects_invalid_schema(tmp_path) -> None:
    from tracing.serialization import TraceLoadError, load_trace_bundle

    store = ArtifactStore(tmp_path)
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload={"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
                  "created": 1234567890, "model": "test",
                  "generation": {}, "trace": {}},
    )
    # Write valid JSON that is not a valid TraceEnvelope (missing required top-level fields)
    bad_payload = '{"not": "a", "valid": "trace"}'
    bad_bytes = bad_payload.encode("utf-8")
    (tmp_path / manifest.path).write_bytes(bad_bytes)
    manifest = manifest.model_copy(update={
        "byte_size": len(bad_bytes),
        "sha256": hashlib.sha256(bad_bytes).hexdigest(),
    })

    with pytest.raises(TraceLoadError, match="does not match"):
        load_trace_bundle(tmp_path, manifest)


# ---------------------------------------------------------------------------
# Trace bundle manifest fields completeness
# ---------------------------------------------------------------------------


def test_trace_bundle_manifest_all_fields(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    payload = {"schema_version": "wllm.trace.v1", "id": "trace_1", "object": "wllm.trace",
               "created": 1234567890, "model": "test",
               "generation": {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
               "trace": {"tokens": {}, "spans": {}, "logprobs": {}, "hidden_states": [], "attentions": []}}
    manifest = store.put_trace_bundle(
        trace_id="trace_1",
        schema_version="wllm.trace.v1",
        payload=payload,
    )

    assert isinstance(manifest.manifest_id, str) and len(manifest.manifest_id) > 0
    assert manifest.object == "wllm.trace_manifest"
    assert manifest.schema_version == "wllm.trace.v1"
    assert manifest.trace_id == "trace_1"
    assert manifest.byte_size >= 0
    assert isinstance(manifest.sha256, str) and len(manifest.sha256) == 64
    assert manifest.path.endswith(".json") and not manifest.path.startswith("/")
    assert isinstance(manifest.created, int) and manifest.created > 0
