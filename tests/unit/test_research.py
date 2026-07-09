from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from research.actmap import ActMapAdapter
from research.eigenscore import EigenScoreAdapter
from research.rauq import RAUQAdapter
from research.token_baselines import TokenBaselineAdapter
from schemas.traces import TokenTrace, TraceData, TraceEnvelope

# Paper-specific adapter names that must not leak into public API surfaces.
_PAPER_ADAPTER_NAMES = {"RAUQ", "rauq", "EigenScore", "eigenscore", "ActMap", "actmap"}

# Public API directories that must remain free of paper-specific terms.
_PUBLIC_API_DIRS = ["schemas", "server"]


def make_trace() -> TraceEnvelope:
    return TraceEnvelope(
        id="trace_1",
        created=0,
        model="fake",
        generation={"id": "cmpl_1", "choices": [], "usage": {}},
        trace=TraceData(tokens=TokenTrace(token_ids=[1, 2, 3], tokens=["a", "b", "c"]), spans={"generated": (1, 3)}),
    )


def test_token_baseline_adapter_on_synthetic_trace() -> None:
    result = TokenBaselineAdapter().run(make_trace())
    assert result.status == "ok"
    assert result.values["token_count"] == 3
    assert result.values["generated_token_count"] == 2


def test_paper_adapters_are_isolated_and_labelled_partial() -> None:
    trace = make_trace()
    for adapter in [EigenScoreAdapter(), RAUQAdapter(), ActMapAdapter()]:
        result = adapter.run(trace)
        assert result.status == "unsupported"
        assert result.warnings


# ---------------------------------------------------------------------------
# No paper-specific terms in public API
# ---------------------------------------------------------------------------


class PaperTermChecker(ast.NodeVisitor):
    """AST visitor that flags paper-specific adapter names in string literals."""

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            for term in _PAPER_ADAPTER_NAMES:
                if term in node.value:
                    self.violations.append((node.lineno, node.value))
        self.generic_visit(node)


def _collect_python_files(directory: Path) -> list[Path]:
    """Recursively collect all .py files under directory, excluding __pycache__."""
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in str(p))


@pytest.mark.parametrize("dir_name", _PUBLIC_API_DIRS)
def test_no_paper_specific_terms_in_public_api_dir(dir_name: str) -> None:
    """Paper-specific adapter names (RAUQ, EigenScore, ActMap) must not appear
    in any source file under the specified public API directory."""
    src_root = Path(__file__).resolve().parent.parent.parent / "src"
    directory = src_root / dir_name
    violations: list[str] = []

    for py_file in _collect_python_files(directory):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        checker = PaperTermChecker()
        checker.visit(tree)
        for lineno, text in checker.violations:
            violations.append(f"{py_file.relative_to(src_root)}:{lineno}: {text!r}")

    assert violations == [], (
        f"Paper-specific terms found in public API directory '{dir_name}':\n"
        + "\n".join(violations)
        + "\n\nPaper-specific adapter names belong in src/research/ only."
    )


def test_no_paper_specific_terms_in_cli() -> None:
    """Paper-specific adapter names must not appear in cli.py."""
    src_root = Path(__file__).resolve().parent.parent.parent / "src"
    cli_path = src_root / "cli.py"
    source = cli_path.read_text(encoding="utf-8")
    violations = [term for term in _PAPER_ADAPTER_NAMES if term in source]
    assert violations == [], (
        f"Paper-specific terms found in cli.py: {violations}. "
        "Paper-specific adapter names belong in src/research/ only."
    )


# ---------------------------------------------------------------------------
# One-line artifact loader verification
# ---------------------------------------------------------------------------


def test_one_line_load_artifact_with_manifest_instance(tmp_path) -> None:
    """load_artifact accepts an ArtifactManifest instance (one-liner)."""
    import numpy as np

    from artifacts import load_artifact
    from artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path)
    tensors = {"a": np.zeros((2, 3), dtype=np.float32)}
    manifest = store.put(trace_id="test", tensors=tensors, format="npz")

    loaded = load_artifact(tmp_path, manifest)
    assert np.array_equal(loaded["a"], tensors["a"])


def test_one_line_load_artifact_with_manifest_dict(tmp_path) -> None:
    """load_artifact accepts a plain dict (one-liner with dict)."""
    import numpy as np

    from artifacts import load_artifact
    from artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path)
    tensors = {"a": np.zeros((2, 3), dtype=np.float32)}
    manifest = store.put(trace_id="test", tensors=tensors, format="npz")

    # One-liner: pass manifest.model_dump() as a plain dict
    loaded = load_artifact(tmp_path, manifest.model_dump(mode="json"))
    assert np.array_equal(loaded["a"], tensors["a"])


def test_one_line_load_trace_bundle_with_manifest(tmp_path) -> None:
    """load_trace_bundle accepts a manifest (one-liner)."""
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.capabilities import default_vllm_capabilities
    from runtime.orchestration import ExtractionOrchestrator
    from schemas.extraction import ExtractRequest
    from tests.unit.test_orchestration import make_inputs
    from tracing.serialization import load_trace_bundle

    request = ExtractRequest.model_validate(
        {"model": "fake", "prompt": "hello", "extract": {"tokens": True}},
    )
    orchestrator = ExtractionOrchestrator(default_vllm_capabilities("fake", "fake"))
    trace = orchestrator.build_trace(
        request,
        make_inputs(),
        limits=ResourceLimits(),
        artifact_store=ArtifactStore(tmp_path),
        persist=True,
    )

    # One-liner: load the trace bundle from its manifest
    loaded = load_trace_bundle(tmp_path, trace.trace_manifest)
    assert loaded.id == trace.id
    assert loaded.trace.tokens.token_ids is not None


# ---------------------------------------------------------------------------
# Dataset workflow example helpers
# ---------------------------------------------------------------------------


def _load_dataset_workflow_module():
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "scripts" / "dataset_workflow.py"
    spec = importlib.util.spec_from_file_location("dataset_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_workflow_read_prompts_returns_valid_entries(tmp_path) -> None:
    workflow = _load_dataset_workflow_module()
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                '{"id": "ok", "prompt": "Explain calibration."}',
                "",
                '{"id": "ok2", "prompt": "Explain uncertainty."}',
            ]
        ),
        encoding="utf-8",
    )

    prompts = workflow.read_prompts(prompt_file)

    assert prompts == [
        {"id": "ok", "prompt": "Explain calibration."},
        {"id": "ok2", "prompt": "Explain uncertainty."},
    ]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "missing"}', "missing non-empty 'prompt' string"),
        ('{"id": "empty", "prompt": ""}', "missing non-empty 'prompt' string"),
        ("not-json", "malformed JSONL entry"),
        ('["not", "an", "object"]', "expected a JSON object"),
    ],
)
def test_dataset_workflow_read_prompts_rejects_bad_records(tmp_path, line: str, message: str) -> None:
    workflow = _load_dataset_workflow_module()
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(line, encoding="utf-8")

    with pytest.raises(workflow.PromptFileError, match=message):
        workflow.read_prompts(prompt_file)


def test_dataset_workflow_read_prompts_rejects_missing_file(tmp_path) -> None:
    workflow = _load_dataset_workflow_module()

    with pytest.raises(workflow.PromptFileError, match="Could not read prompts file"):
        workflow.read_prompts(tmp_path / "missing.jsonl")


def test_dataset_workflow_read_prompts_rejects_invalid_utf8(tmp_path) -> None:
    workflow = _load_dataset_workflow_module()
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_bytes(b"\x80\x81\x82")

    with pytest.raises(workflow.PromptFileError, match="not valid UTF-8"):
        workflow.read_prompts(prompt_file)


def test_dataset_workflow_main_reports_prompt_file_errors(tmp_path, monkeypatch, capsys) -> None:
    workflow = _load_dataset_workflow_module()
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(sys, "argv", ["dataset_workflow.py", "--prompts", str(missing)])

    rc = workflow.main()

    captured = capsys.readouterr()
    assert rc == 1
    assert "Prompt file error:" in captured.err
    assert str(missing) in captured.err


def test_dataset_workflow_main_preserves_rows_when_adapter_fails(tmp_path, monkeypatch) -> None:
    workflow = _load_dataset_workflow_module()
    prompt_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "results.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                json.dumps({"id": "ok", "prompt": "first prompt"}),
                json.dumps({"id": "boom", "prompt": "second prompt"}),
            ]
        ),
        encoding="utf-8",
    )

    class Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    class AdapterResult:
        name = "token_baselines"
        status = "ok"
        values = {"token_count": 3}

    client = Client()
    adapter_calls = 0

    def fake_extract_trace(client_arg, **kwargs):
        assert client_arg is client
        return {"trace_manifest": {"path": f"{kwargs['prompt']}.json"}, "artifacts": []}

    def fake_run_adapter(trace):
        nonlocal adapter_calls
        adapter_calls += 1
        if adapter_calls == 2:
            raise RuntimeError("adapter boom")
        return AdapterResult()

    monkeypatch.setattr(workflow, "_get_client", lambda: client)
    monkeypatch.setattr(workflow, "extract_trace", fake_extract_trace)
    monkeypatch.setattr(workflow, "load_trace_and_artifacts", lambda artifact_dir, response: (make_trace(), {}))
    monkeypatch.setattr(workflow, "run_adapter", fake_run_adapter)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset_workflow.py", "--prompts", str(prompt_file), "--output", str(output_file), "--model", "fake"],
    )

    rc = workflow.main()

    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert rc == 0
    assert client.closed
    assert [row["id"] for row in rows] == ["ok", "boom"]
    assert rows[0]["adapter_status"] == "ok"
    assert rows[1]["trace_id"] == "trace_1"
    assert rows[1]["error"] == "adapter: adapter boom"


def test_dataset_workflow_extract_trace_builds_expected_request_body() -> None:
    workflow = _load_dataset_workflow_module()

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"trace_manifest": {"path": "trace.json"}, "artifacts": []}

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def post(self, path: str, json: dict[str, object]) -> Response:
            self.calls.append((path, json))
            return Response()

    client = Client()

    response = workflow.extract_trace(
        client,
        prompt="hello",
        model="fake",
        max_tokens=8,
        include_logprobs=True,
        include_hidden_states=True,
        top_k=3,
    )

    assert response["artifacts"] == []
    assert client.calls[0][0] == "/traces"
    body = client.calls[0][1]
    assert body["model"] == "fake"
    assert body["prompt"] == "hello"
    assert body["max_tokens"] == 8
    extract = body["extract"]
    assert extract["tokens"] is True
    assert extract["logprobs"] == {"top_k": 3, "include_prompt": True}
    assert extract["hidden_states"] == [{"layers": "middle", "positions": "last_generated"}]
    assert extract["artifacts"] == {"format": "npz", "include": ["logprobs", "hidden_states"]}


def test_dataset_workflow_extract_trace_can_request_tokens_only() -> None:
    workflow = _load_dataset_workflow_module()

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"trace_manifest": {"path": "trace.json"}, "artifacts": []}

    class Client:
        def __init__(self) -> None:
            self.body: dict[str, object] | None = None

        def post(self, path: str, json: dict[str, object]) -> Response:
            del path
            self.body = json
            return Response()

    client = Client()
    workflow.extract_trace(client, prompt="hello", model="fake", max_tokens=8, include_logprobs=False)

    assert client.body is not None
    assert client.body["extract"] == {"tokens": True}


def test_dataset_workflow_error_message_prefers_openai_error_envelope() -> None:
    workflow = _load_dataset_workflow_module()

    class Response:
        status_code = 413

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"code": "extraction_limit_exceeded", "message": "too large"}}

    class HTTPError(Exception):
        response = Response()

    assert workflow.error_message(HTTPError("raw")) == "extraction_limit_exceeded: too large"


def test_dataset_workflow_build_result_uses_generated_span() -> None:
    workflow = _load_dataset_workflow_module()

    class AdapterResult:
        name = "token_baselines"
        status = "ok"
        values = {"token_count": 3}

    result = workflow.build_result(
        {"id": "p1", "prompt": "hello"},
        {"trace_manifest": {}},
        make_trace(),
        {"art.npz": {}},
        AdapterResult(),
    )

    assert result["id"] == "p1"
    assert result["trace_id"] == "trace_1"
    assert result["generated_token_count"] == 2
    assert result["artifact_count"] == 1


# ---------------------------------------------------------------------------
# Generic feature helpers (chosen_logprobs, hidden_states_matrix)
# ---------------------------------------------------------------------------


def test_chosen_logprobs_from_trace() -> None:
    from research.features import chosen_logprobs

    # Build a minimal trace with logprobs structure
    trace = make_trace()
    # Manually attach logprobs in the shape produced by orchestration
    trace.trace.logprobs = {
        "generated": [
            {"token_id": 12, "token": "c", "logprob": -0.2, "top_logprobs": []},
            {"token_id": 13, "token": "d", "logprob": -0.1, "top_logprobs": []},
        ],
        "prompt": [
            {"token_id": 10, "token": "a", "logprob": -0.5, "top_logprobs": []},
        ],
    }

    res = chosen_logprobs(trace)
    assert "generated" in res
    assert res["generated"] == [-0.2, -0.1]

    res_p = chosen_logprobs(trace, include_prompt=True)
    assert "prompt" in res_p
    assert res_p["prompt"] == [-0.5]


def test_hidden_states_matrix_best_effort() -> None:
    from research.features import hidden_states_matrix

    trace = make_trace()
    # Attach a simple hidden record (inline data case)
    class Rec:
        layers = [5]
        positions = "generated"
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # 2 pos x 3 dim

    trace.trace.hidden_states = [Rec()]

    mats = hidden_states_matrix(trace, layer=5)
    assert "generated" in mats or mats  # best effort may key it
    # At minimum it should not crash and return something array-like if data present
    if mats:
        for v in mats.values():
            assert hasattr(v, "shape")


def test_last_token_hidden_helper() -> None:
    from research.features import last_token_hidden
    import numpy as np

    trace = make_trace()
    class Rec:
        layers = [5]
        positions = "last_generated"
        data = np.array([[0.1, 0.2], [0.3, 0.4]])  # last row is the one

    trace.trace.hidden_states = [Rec()]
    vec = last_token_hidden(trace, layer=5)
    assert vec is not None
    assert vec.shape == (2,)
    assert abs(float(vec[0]) - 0.3) < 1e-5
