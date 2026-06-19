from __future__ import annotations

from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class ActMapAdapter:
    name = "actmap"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del trace, options
        return ResearchResult(
            name=self.name,
            status="unsupported",
            warnings=["ActMap requires a separately validated adapter over generic activation artifacts."],
        )
