from __future__ import annotations

from contextvars import ContextVar

active_trace_id: ContextVar[str | None] = ContextVar("active_trace_id", default=None)
