from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from extractors.planning import ResourceLimits
from schemas.openai import ChatMessage

EXTRACTION_SCHEMA_VERSION = "wllm.extraction.v1"

LayerSelector: TypeAlias = int | list[int] | Literal["all", "middle", "middle_third"]
PositionSelector: TypeAlias = int | list[int] | Literal["prompt", "generated", "last", "last_generated"]
AttentionKeySelector: TypeAlias = PositionSelector | Literal["previous_token"]
HeadSelector: TypeAlias = int | list[int] | Literal["all"]
PoolOp: TypeAlias = Literal["mean", "max", "last"] | None


class LogprobsExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int | None = Field(default=None, ge=1)
    include_prompt: bool = False
    entropy: bool = False
    raw_logits: bool = False
    allow_approximate_entropy: bool = False


class HiddenStateExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: LayerSelector
    positions: PositionSelector
    pool: PoolOp = None


class AttentionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: LayerSelector
    heads: HeadSelector = "all"
    query_positions: PositionSelector
    key_positions: AttentionKeySelector


class ArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["npz", "pt"] = "npz"
    include: list[Literal["tokens", "logprobs", "hidden_states", "attentions"]] = Field(default_factory=list)
    allow_large: bool = False


class ExtractionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: bool = False
    logprobs: LogprobsExtraction | None = None
    hidden_states: list[HiddenStateExtraction] = Field(default_factory=list)
    attentions: list[AttentionExtraction] = Field(default_factory=list)
    artifacts: ArtifactRequest | None = None


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] | None = None
    prompt: str | None = None
    max_tokens: int = Field(default=16, ge=0)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False
    extract: ExtractionSpec = Field(default_factory=ExtractionSpec)

    @model_validator(mode="after")
    def exactly_one_prompt_source(self) -> "ExtractRequest":
        has_messages = self.messages is not None
        has_prompt = self.prompt is not None
        if has_messages == has_prompt:
            raise ValueError("exactly one of messages or prompt must be provided")
        return self


class TraceRequest(ExtractRequest):
    pass


class ExtractionSchemaResponse(BaseModel):
    schema_version: Literal["wllm.extraction.v1"] = EXTRACTION_SCHEMA_VERSION
    request_schema: dict[str, Any]
    selectors: dict[str, Any]
    limits: ResourceLimits
    capabilities: Any


def extraction_schema_payload(limits: ResourceLimits, capabilities: Any) -> ExtractionSchemaResponse:
    selectors = {
        "layers": {
            "types": ["integer", "integer[]", "all", "middle", "middle_third"],
            "semantics": {
                "positive": "zero-based transformer-block indexes",
                "negative": "resolved from the end after model layer count is known",
                "middle": "(num_layers - 1) // 2",
                "middle_third": "floor(num_layers / 3) through ceil(2 * num_layers / 3) - 1",
            },
        },
        "positions": {
            "types": ["integer", "integer[]", "prompt", "generated", "last", "last_generated"],
            "semantics": {
                "prompt": "[0, prompt_token_count)",
                "generated": "[prompt_token_count, total_token_count)",
                "negative": "resolved from the final combined token sequence",
            },
        },
        "attention_key_positions": {
            "additional": ["previous_token"],
            "previous_token": "for each selected query q, select key q - 1 when q > 0",
        },
        "pooling": [None, "mean", "max", "last"],
    }
    return ExtractionSchemaResponse(
        request_schema=ExtractRequest.model_json_schema(),
        selectors=selectors,
        limits=limits,
        capabilities=capabilities,
    )
