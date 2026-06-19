from __future__ import annotations

from typing import Any

from server.errors import UnsupportedExtractionError


def attentions_unavailable(details: dict[str, Any] | None = None) -> None:
    raise UnsupportedExtractionError(
        "Selected attention weights are not exposed by the active vLLM serving path.",
        code="attention_weights_unavailable",
        param="extract.attentions",
        details=details,
    )
