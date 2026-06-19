from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from schemas.traces import TraceEnvelope


class ResearchResult(BaseModel):
    name: str
    status: str
    values: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ResearchAdapter(Protocol):
    name: str

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        ...
