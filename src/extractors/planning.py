from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from extractors.selectors import (
    SelectorValidationError,
    normalize_head_selector,
    normalize_layer_selector,
    normalize_position_selector,
    resolve_attention_key_positions,
)
from server.errors import InvalidRequestError, ResourceLimitError


class ResourceLimits(BaseModel):
    max_top_k: int = Field(default=50, ge=1)
    max_selected_layers: int = Field(default=8, ge=1)
    max_selected_heads: int = Field(default=32, ge=1)
    max_selected_positions: int = Field(default=256, ge=1)
    max_inline_tensor_bytes: int = Field(default=1_000_000, ge=1)
    max_total_captured_tensor_bytes: int = Field(default=64_000_000, ge=1)
    max_artifact_bytes: int = Field(default=256_000_000, ge=1)
    large_extraction_enabled: bool = False


class RuntimeTopology(BaseModel):
    num_layers: int = Field(gt=0)
    num_attention_heads: int | None = Field(default=None, gt=0)
    hidden_size: int | None = Field(default=None, gt=0)


class ExtractionPlan(BaseModel):
    logprobs: dict[str, Any] | None = None
    hidden_states: list[dict[str, Any]] = Field(default_factory=list)
    attentions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_selectors: dict[str, Any] = Field(default_factory=dict)


def compile_extraction_plan(
    spec: Any,
    *,
    num_layers: int,
    prompt_token_count: int,
    generated_token_count: int,
    limits: ResourceLimits,
    num_heads: int | None = None,
) -> ExtractionPlan:
    try:
        return _compile(spec, num_layers, prompt_token_count, generated_token_count, limits, num_heads)
    except SelectorValidationError as exc:
        raise InvalidRequestError(str(exc), code="invalid_selector", param=exc.param) from exc


def validate_pre_generation_selectors(
    spec: Any,
    *,
    topology: RuntimeTopology,
    limits: ResourceLimits,
) -> None:
    try:
        for index, hidden in enumerate(spec.hidden_states):
            layers = normalize_layer_selector(hidden.layers, topology.num_layers)
            _check_or_require_large(
                spec,
                tensor_name="hidden_states",
                name="layers",
                values=layers,
                limit=limits.max_selected_layers,
                param=f"extract.hidden_states[{index}].layers",
                limits=limits,
            )
        for index, attention in enumerate(spec.attentions):
            layers = normalize_layer_selector(attention.layers, topology.num_layers)
            heads = normalize_head_selector(attention.heads, topology.num_attention_heads)
            _check_or_require_large(
                spec,
                tensor_name="attentions",
                name="layers",
                values=layers,
                limit=limits.max_selected_layers,
                param=f"extract.attentions[{index}].layers",
                limits=limits,
            )
            if heads != "all":
                _check_or_require_large(
                    spec,
                    tensor_name="attentions",
                    name="heads",
                    values=heads,
                    limit=limits.max_selected_heads,
                    param=f"extract.attentions[{index}].heads",
                    limits=limits,
                )
    except SelectorValidationError as exc:
        raise InvalidRequestError(str(exc), code="invalid_selector", param=exc.param) from exc


def _compile(
    spec: Any,
    num_layers: int,
    prompt_token_count: int,
    generated_token_count: int,
    limits: ResourceLimits,
    num_heads: int | None,
) -> ExtractionPlan:
    plan = ExtractionPlan()
    if spec.logprobs is not None:
        top_k = spec.logprobs.top_k
        if top_k is not None and top_k > limits.max_top_k:
            raise ResourceLimitError(
                f"Requested top_k={top_k} exceeds max_top_k={limits.max_top_k}.",
                param="extract.logprobs.top_k",
                details={"requested": top_k, "limit": limits.max_top_k},
            )
        plan.logprobs = spec.logprobs.model_dump(mode="json")

    hidden_metadata = []
    for index, hidden in enumerate(spec.hidden_states):
        layers = normalize_layer_selector(hidden.layers, num_layers)
        positions = normalize_position_selector(
            hidden.positions,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
        )
        _check_or_require_large(
            spec,
            tensor_name="hidden_states",
            name="layers",
            values=layers,
            limit=limits.max_selected_layers,
            param=f"extract.hidden_states[{index}].layers",
            limits=limits,
        )
        _check_or_require_large(
            spec,
            tensor_name="hidden_states",
            name="positions",
            values=positions,
            limit=limits.max_selected_positions,
            param=f"extract.hidden_states[{index}].positions",
            limits=limits,
        )
        if len(layers) == num_layers and len(positions) == prompt_token_count + generated_token_count:
            _require_large_extraction(
                spec,
                tensor_name="hidden_states",
                param=f"extract.hidden_states[{index}]",
                limits=limits,
                reason="Full hidden-state dumps must be artifact-backed and explicitly enabled.",
            )
        item = {"layers": layers, "positions": positions, "pool": hidden.pool, "capture_mode": hidden.capture_mode}
        hidden_metadata.append(item)
        plan.hidden_states.append(item)

    attention_metadata = []
    for index, attention in enumerate(spec.attentions):
        layers = normalize_layer_selector(attention.layers, num_layers)
        heads = normalize_head_selector(attention.heads, num_heads)
        query_positions = normalize_position_selector(
            attention.query_positions,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
        )
        key_positions = resolve_attention_key_positions(
            attention.key_positions,
            query_positions=query_positions,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
        )
        _check_or_require_large(
            spec,
            tensor_name="attentions",
            name="layers",
            values=layers,
            limit=limits.max_selected_layers,
            param=f"extract.attentions[{index}].layers",
            limits=limits,
        )
        if heads != "all":
            _check_or_require_large(
                spec,
                tensor_name="attentions",
                name="heads",
                values=heads,
                limit=limits.max_selected_heads,
                param=f"extract.attentions[{index}].heads",
                limits=limits,
            )
        _check_or_require_large(
            spec,
            tensor_name="attentions",
            name="positions",
            values=query_positions,
            limit=limits.max_selected_positions,
            param=f"extract.attentions[{index}].query_positions",
            limits=limits,
        )
        if len(layers) == num_layers and heads == "all" and len(
                query_positions) == prompt_token_count + generated_token_count:
            _require_large_extraction(
                spec,
                tensor_name="attentions",
                param=f"extract.attentions[{index}]",
                limits=limits,
                reason="Full attention dumps must be artifact-backed and explicitly enabled.",
            )
        item = {"layers": layers, "heads": heads, "query_positions": query_positions, "key_positions": key_positions}
        attention_metadata.append(item)
        plan.attentions.append(item)

    plan.resolved_selectors = {"hidden_states": hidden_metadata, "attentions": attention_metadata}
    return plan


def _check_or_require_large(
    spec: Any,
    *,
    tensor_name: str,
    name: str,
    values: list[int],
    limit: int,
    param: str,
    limits: ResourceLimits,
) -> None:
    if len(values) <= limit:
        return
    _require_large_extraction(
        spec,
        tensor_name=tensor_name,
        param=param,
        limits=limits,
        reason=f"Selected {len(values)} {name}, which exceeds the default limit of {limit}.",
    )


def _require_large_extraction(
    spec: Any,
    *,
    tensor_name: str,
    param: str,
    limits: ResourceLimits,
    reason: str,
) -> None:
    artifacts = spec.artifacts
    include = set(artifacts.include) if artifacts is not None else set()
    details = {
        "reason": reason,
        "required_artifact_include": tensor_name,
        "request_allow_large": bool(artifacts and artifacts.allow_large),
        "server_large_extraction_enabled": limits.large_extraction_enabled,
    }
    if artifacts is None or tensor_name not in include:
        raise ResourceLimitError(
            f"{reason} Request an artifact including {tensor_name!r}.",
            param=param,
            details=details,
        )
    if not artifacts.allow_large:
        raise ResourceLimitError(
            f"{reason} Set extract.artifacts.allow_large=true.",
            param="extract.artifacts.allow_large",
            details=details,
        )
    if not limits.large_extraction_enabled:
        raise ResourceLimitError(
            f"{reason} The server has not enabled large extraction.",
            param=param,
            details=details,
        )
