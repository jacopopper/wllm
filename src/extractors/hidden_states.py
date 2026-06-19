from __future__ import annotations

from typing import Any

from server.errors import UnsupportedExtractionError


def hidden_states_unavailable(details: dict[str, Any] | None = None) -> None:
    raise UnsupportedExtractionError(
        "Hidden-state extraction is not available for the active vLLM model or worker configuration.",
        code="hidden_states_unavailable",
        param="extract.hidden_states",
        details=details,
    )
