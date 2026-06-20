from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CapabilityState = Literal["supported", "unsupported", "conditional"]


class Capability(BaseModel):
    state: CapabilityState
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RuntimeCapabilities(BaseModel):
    model: str
    vllm_version: str | None = None
    execution_mode: str = "vllm"
    attention_backend: str | None = None
    token_ids: Capability
    token_logprobs: Capability
    prompt_logprobs: Capability
    top_k_logprobs: Capability
    top_k_logits: Capability
    exact_entropy: Capability
    hidden_states: Capability
    attentions: Capability
    npz_artifacts: Capability
    pt_artifacts: Capability

    def as_metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def default_vllm_capabilities(
    model: str,
    vllm_version: str | None = None,
    *,
    attention_backend: str | None = None,
    supported_runner_types: list[str] | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    online_hidden_states: bool = False,
) -> RuntimeCapabilities:
    replay_available = (
        tensor_parallel_size == 1
        and gpu_memory_utilization <= 0.5
        and (supported_runner_types is None or "pooling" in supported_runner_types)
    )
    online_available = online_hidden_states and tensor_parallel_size == 1
    capture_modes = []
    if replay_available:
        capture_modes.append("replay")
    if online_available:
        capture_modes.append("online")

    if tensor_parallel_size != 1:
        hidden_states = Capability(
            state="unsupported",
            reason="Scoped hidden-state capture currently supports only tensor_parallel_size=1.",
            details={"tensor_parallel_size": tensor_parallel_size},
        )
    elif not capture_modes and gpu_memory_utilization > 0.5:
        hidden_states = Capability(
            state="unsupported",
            reason=(
                "Scoped hidden-state capture starts an isolated vLLM pooling runner "
                "in addition to the generation runner; configure "
                "gpu_memory_utilization <= 0.5 for replay extraction or enable "
                "online hidden-state capture."
            ),
            details={
                "gpu_memory_utilization": gpu_memory_utilization,
                "maximum_supported_for_replay_hidden_states": 0.5,
                "online_hidden_states_enabled": online_hidden_states,
            },
        )
    elif not capture_modes and supported_runner_types is not None and "pooling" not in supported_runner_types:
        hidden_states = Capability(
            state="unsupported",
            reason=(
                "The loaded vLLM model configuration does not advertise the "
                "pooling runner required for replay hidden-state extraction, "
                "and online hidden-state capture is not enabled."
            ),
            details={"supported_runner_types": supported_runner_types,
                     "online_hidden_states_enabled": online_hidden_states},
        )
    else:
        hidden_states = Capability(
            state="conditional",
            reason=(
                "Selected transformer-block token hidden states require "
                "temporary scoped module hooks and tensor_parallel_size=1. "
                "Replay mode uses an isolated vLLM pooling runner; online "
                "mode uses an opt-in eager in-process generation runner."
            ),
            details={
                "backend": "vllm_scoped_hooks",
                "capture_modes": capture_modes,
                "supported_runner_types": supported_runner_types,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "online_hidden_states_enabled": online_hidden_states,
                "capture_site": "transformer_block_output",
            },
        )
    return RuntimeCapabilities(
        model=model,
        vllm_version=vllm_version,
        attention_backend=attention_backend,
        token_ids=Capability(state="supported"),
        token_logprobs=Capability(state="conditional", reason="Requires requesting logprobs from vLLM SamplingParams."),
        prompt_logprobs=Capability(
            state="conditional",
            reason="Requires extract.logprobs.include_prompt=true and vLLM SamplingParams.prompt_logprobs support.",
        ),
        top_k_logprobs=Capability(state="conditional", reason="Bounded by the configured maximum top_k."),
        top_k_logits=Capability(
            state="unsupported",
            reason="The public vLLM generation output exposes normalized logprobs, not raw logits.",
        ),
        exact_entropy=Capability(
            state="unsupported",
            reason="The public vLLM generation output does not expose the complete token distribution.",
        ),
        hidden_states=hidden_states,
        attentions=Capability(
            state="unsupported",
            reason="Attention weights are not exposed by fused attention backends through the public path.",
            details={"attention_backend": attention_backend} if attention_backend is not None else {},
        ),
        npz_artifacts=Capability(state="supported"),
        pt_artifacts=Capability(state="conditional", reason="Requires torch to be installed and configured."),
    )
