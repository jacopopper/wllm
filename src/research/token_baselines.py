from __future__ import annotations

from research.base import ResearchResult
from schemas.traces import TraceEnvelope


class TokenBaselineAdapter:
    name = "token_baselines"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        del options
        tokens = trace.trace.tokens
        token_count = len(tokens.token_ids) if hasattr(tokens, "token_ids") else 0
        generated_span = trace.trace.spans.get("generated", (0, 0))
        return ResearchResult(
            name=self.name,
            status="ok",
            values={
                "token_count": token_count,
                "generated_token_count": max(0, generated_span[1] - generated_span[0]),
            },
        )
