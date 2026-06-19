from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal


class SelectorValidationError(ValueError):
    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


def _dedupe(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_layer_selector(selector: int | list[int] | str, num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise SelectorValidationError("num_layers must be positive", param="layers")
    if isinstance(selector, int):
        values = [_resolve_index(selector, num_layers, "layers")]
    elif isinstance(selector, list):
        values = [_resolve_index(value, num_layers, "layers") for value in selector]
    elif selector == "all":
        values = list(range(num_layers))
    elif selector == "middle":
        values = [(num_layers - 1) // 2]
    elif selector == "middle_third":
        start = math.floor(num_layers / 3)
        stop = math.ceil(2 * num_layers / 3)
        values = list(range(start, stop))
    else:
        raise SelectorValidationError(f"Unknown layer selector {selector!r}.", param="layers")
    return _dedupe(values)


def normalize_head_selector(selector: int | list[int] | Literal["all"], num_heads: int | None) -> list[int] | Literal["all"]:
    if selector == "all":
        return "all"
    if num_heads is None:
        return [selector] if isinstance(selector, int) else _dedupe(selector)
    if isinstance(selector, int):
        return [_resolve_index(selector, num_heads, "heads")]
    return _dedupe(_resolve_index(value, num_heads, "heads") for value in selector)


def normalize_position_selector(
    selector: int | list[int] | str,
    *,
    prompt_token_count: int,
    generated_token_count: int,
) -> list[int]:
    if prompt_token_count < 0 or generated_token_count < 0:
        raise SelectorValidationError("token counts must be non-negative", param="positions")
    total = prompt_token_count + generated_token_count
    if isinstance(selector, int):
        values = [_resolve_index(selector, total, "positions")]
    elif isinstance(selector, list):
        values = [_resolve_index(value, total, "positions") for value in selector]
    elif selector == "prompt":
        values = list(range(prompt_token_count))
    elif selector == "generated":
        values = list(range(prompt_token_count, total))
    elif selector == "last":
        if total == 0:
            raise SelectorValidationError("last position is invalid for an empty token sequence", param="positions")
        values = [total - 1]
    elif selector == "last_generated":
        if generated_token_count == 0:
            raise SelectorValidationError("last_generated is invalid when no token was generated", param="positions")
        values = [total - 1]
    else:
        raise SelectorValidationError(f"Unknown position selector {selector!r}.", param="positions")
    return _dedupe(values)


def resolve_attention_key_positions(
    selector: int | list[int] | str,
    *,
    query_positions: list[int],
    prompt_token_count: int,
    generated_token_count: int,
) -> dict[int, list[int]]:
    if selector == "previous_token":
        return {query: [query - 1] for query in query_positions if query > 0}
    keys = normalize_position_selector(
        selector,
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
    )
    return {query: keys for query in query_positions}


def _resolve_index(index: int, size: int, param: str) -> int:
    if size <= 0:
        raise SelectorValidationError(f"{param} index {index} is invalid for an empty sequence.", param=param)
    resolved = index if index >= 0 else size + index
    if resolved < 0 or resolved >= size:
        raise SelectorValidationError(
            f"{param} index {index} resolves to {resolved}, outside [0, {size}).",
            param=param,
        )
    return resolved
