from __future__ import annotations

import json

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
    def fail_save(path, tensors):
        del tensors
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
