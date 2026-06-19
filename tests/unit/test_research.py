from __future__ import annotations

from research.actmap import ActMapAdapter
from research.eigenscore import EigenScoreAdapter
from research.rauq import RAUQAdapter
from research.token_baselines import TokenBaselineAdapter
from schemas.traces import TokenTrace, TraceData, TraceEnvelope


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
