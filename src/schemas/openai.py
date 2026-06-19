from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class SamplingFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = Field(default=16, ge=0)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: bool | int | None = None
    stream: bool = False


class ChatCompletionRequest(SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)


class CompletionRequest(SamplingFields):
    prompt: str | list[str]

    @model_validator(mode="after")
    def prompt_not_empty(self) -> "CompletionRequest":
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        if not prompts or any(prompt == "" for prompt in prompts):
            raise ValueError("prompt must contain at least one non-empty string")
        return self


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: Any | None = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None
    logprobs: Any | None = None


class OpenAIListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[dict[str, Any]]
