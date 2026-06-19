from __future__ import annotations

import pytest

from extractors.selectors import (
    SelectorValidationError,
    normalize_layer_selector,
    normalize_position_selector,
    resolve_attention_key_positions,
)


def test_layer_selector_semantics() -> None:
    assert normalize_layer_selector(0, 12) == [0]
    assert normalize_layer_selector(-1, 12) == [11]
    assert normalize_layer_selector([0, -1, 0], 12) == [0, 11]
    assert normalize_layer_selector("all", 4) == [0, 1, 2, 3]
    assert normalize_layer_selector("middle", 12) == [5]
    assert normalize_layer_selector("middle_third", 12) == [4, 5, 6, 7]


def test_layer_selector_out_of_range() -> None:
    with pytest.raises(SelectorValidationError):
        normalize_layer_selector(12, 12)


def test_position_selector_semantics() -> None:
    assert normalize_position_selector("prompt", prompt_token_count=3, generated_token_count=2) == [0, 1, 2]
    assert normalize_position_selector("generated", prompt_token_count=3, generated_token_count=2) == [3, 4]
    assert normalize_position_selector("last", prompt_token_count=3, generated_token_count=2) == [4]
    assert normalize_position_selector("last_generated", prompt_token_count=3, generated_token_count=2) == [4]
    assert normalize_position_selector([-1, 0, -1], prompt_token_count=3, generated_token_count=2) == [4, 0]


def test_last_generated_requires_generation() -> None:
    with pytest.raises(SelectorValidationError):
        normalize_position_selector("last_generated", prompt_token_count=3, generated_token_count=0)


def test_previous_token_attention_resolution() -> None:
    assert resolve_attention_key_positions(
        "previous_token",
        query_positions=[0, 2, 4],
        prompt_token_count=3,
        generated_token_count=2,
    ) == {2: [1], 4: [3]}
