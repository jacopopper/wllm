from __future__ import annotations

from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class LinearProbeAdapter:
    name = "linear_probe"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del trace, options
        return ResearchResult(
            name=self.name,
            status="unsupported",
            warnings=["Linear probes require caller-provided probe weights and hidden-state tensors."],
        )
