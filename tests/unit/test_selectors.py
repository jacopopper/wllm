from __future__ import annotations

import pytest

from extractors.selectors import (
    SelectorValidationError,
    normalize_head_selector,
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


# --- Head selector normalization ---


def test_head_selector_all_returns_literal_all() -> None:
    assert normalize_head_selector("all", num_heads=32) == "all"
    assert normalize_head_selector("all", num_heads=None) == "all"


def test_head_selector_integer_resolves() -> None:
    assert normalize_head_selector(0, num_heads=32) == [0]
    assert normalize_head_selector(3, num_heads=32) == [3]
    assert normalize_head_selector(-1, num_heads=32) == [31]
    assert normalize_head_selector([0, 2, -1], num_heads=32) == [0, 2, 31]


def test_head_selector_out_of_range_rejected() -> None:
    with pytest.raises(SelectorValidationError):
        normalize_head_selector(32, num_heads=32)
    with pytest.raises(SelectorValidationError):
        normalize_head_selector(-33, num_heads=32)
    with pytest.raises(SelectorValidationError):
        normalize_head_selector([0, 32], num_heads=32)


def test_head_selector_without_num_heads_passthrough() -> None:
    assert normalize_head_selector(5, num_heads=None) == [5]
    assert normalize_head_selector([1, 2, 3], num_heads=None) == [1, 2, 3]


# --- Negative position selector resolution ---


def test_negative_position_selector_resolution() -> None:
    assert normalize_position_selector(-1, prompt_token_count=3, generated_token_count=2) == [4]
    assert normalize_position_selector(-2, prompt_token_count=3, generated_token_count=2) == [3]
    assert normalize_position_selector(-4, prompt_token_count=3, generated_token_count=2) == [1]


def test_negative_position_selector_out_of_range() -> None:
    with pytest.raises(SelectorValidationError):
        normalize_position_selector(-6, prompt_token_count=3, generated_token_count=2)
    with pytest.raises(SelectorValidationError):
        normalize_position_selector(5, prompt_token_count=3, generated_token_count=2)
