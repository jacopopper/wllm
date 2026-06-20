from __future__ import annotations

import ast
from pathlib import Path

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
