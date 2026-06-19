from __future__ import annotations

from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class RAUQAdapter:
    name = "rauq"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del trace, options
        return ResearchResult(
            name=self.name,
            status="unsupported",
            warnings=["RAUQ is not implemented; use generic trace tensors to build a verified adapter."],
        )
