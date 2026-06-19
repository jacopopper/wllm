from __future__ import annotations

from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class EigenScoreAdapter:
    name = "eigenscore"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del trace, options
        return ResearchResult(
            name=self.name,
            status="unsupported",
            warnings=["EigenScore requires hidden-state tensors from the generic trace layer."],
        )
