from __future__ import annotations

import pytest
from pydantic import ValidationError

from extractors.planning import ResourceLimits, compile_extraction_plan
from schemas.extraction import ExtractionSpec
from server.errors import ResourceLimitError


def test_schema_parsing_and_plan_resolution() -> None:
    spec = ExtractionSpec.model_validate(
        {
            "tokens": True,
            "logprobs": {"top_k": 10, "entropy": False},
            "hidden_states": [{"layers": [0, -1, 0], "positions": "last_generated", "pool": None}],
            "attentions": [
                {
                    "layers": "middle_third",
                    "heads": [0, 1],
                    "query_positions": "generated",
                    "key_positions": "previous_token",
                }
            ],
        }
    )
    plan = compile_extraction_plan(
        spec,
        num_layers=12,
        prompt_token_count=3,
        generated_token_count=2,
        limits=ResourceLimits(max_selected_layers=8),
        num_heads=8,
    )
    assert plan.hidden_states[0]["layers"] == [0, 11]
    assert plan.hidden_states[0]["positions"] == [4]
    assert plan.attentions[0]["layers"] == [4, 5, 6, 7]
    assert plan.attentions[0]["key_positions"] == {3: [2], 4: [3]}


def test_top_k_limit_rejection() -> None:
    spec = ExtractionSpec.model_validate({"logprobs": {"top_k": 99}})
    with pytest.raises(ResourceLimitError):
        compile_extraction_plan(
            spec,
            num_layers=1,
            prompt_token_count=1,
            generated_token_count=1,
            limits=ResourceLimits(max_top_k=4),
        )


def test_logprobs_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExtractionSpec.model_validate({"logprobs": {"top_k": 0}})
