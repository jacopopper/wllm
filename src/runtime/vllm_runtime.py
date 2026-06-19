from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from artifacts.store import ArtifactStore
from extractors.collectors import CollectorRegistry
from extractors.planning import ExtractionPlan, ResourceLimits, RuntimeTopology, compile_extraction_plan
from runtime.capabilities import RuntimeCapabilities, default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator, GeneratedTraceInputs, LogprobCandidate
from runtime.vllm_compat import (
    capture_pooling_hidden_states,
    extract_attention_backend,
    extract_model_topology,
    extract_supported_runner_types,
    import_vllm,
    pooling_runner_environment,
    pooling_runner_kwargs,
    supported_kwargs,
    supported_parameter_names,
)
from schemas.extraction import ExtractRequest
from schemas.openai import ChatCompletionRequest, CompletionRequest
from schemas.traces import TraceEnvelope
from server.errors import InvalidRequestError, ResourceLimitError, RuntimeUnavailableError, UnsupportedExtractionError


@dataclass(frozen=True)
class VLLMRuntimeConfig:
    model: str
    served_model_name: str | None = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    tokenizer: str | None = None
    seed: int | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False


class VLLMRuntime:
    def __init__(self, config: VLLMRuntimeConfig) -> None:
        self.config = config
        self._imports: Any | None = None
        self._llm: Any | None = None
        self._capabilities: RuntimeCapabilities | None = None
        self._collector_registry: CollectorRegistry | None = None
        self._topology: RuntimeTopology | None = None
        self._attention_backend: str | None = None
        self._supported_runner_types: list[str] | None = None
        self._pooling_llm: Any | None = None
        self._load_lock = threading.RLock()
        self._pooling_lock = threading.RLock()

    @property
    def served_model_name(self) -> str:
        return self.config.served_model_name or self.config.model

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with self._load_lock:
            if self._llm is not None:
                return
            if self.config.local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            imports = import_vllm()
            kwargs = supported_kwargs(
                imports.LLM,
                {
                    "model": self.config.model,
                    "dtype": self.config.dtype,
                    "tensor_parallel_size": self.config.tensor_parallel_size,
                    "gpu_memory_utilization": self.config.gpu_memory_utilization,
                    "max_model_len": self.config.max_model_len,
                    "tokenizer": self.config.tokenizer,
                    "trust_remote_code": self.config.trust_remote_code,
                },
            )
            try:
                llm = imports.LLM(**kwargs)
                topology = extract_model_topology(llm)
                attention_backend = extract_attention_backend(llm, imports.module)
                supported_runner_types = extract_supported_runner_types(llm)
                capabilities = default_vllm_capabilities(
                    self.config.model,
                    imports.version,
                    attention_backend=attention_backend,
                    supported_runner_types=supported_runner_types,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                )
            except Exception as exc:
                details = {"exception": repr(exc), "model": self.config.model}
                if self.config.local_files_only:
                    details["local_files_only"] = True
                    message = (
                        "vLLM could not initialize the requested local model. "
                        "Because --local-files-only is set, wllm will not download missing files."
                    )
                else:
                    message = "vLLM could not initialize the requested model."
                raise RuntimeUnavailableError(message, code="vllm_initialization_failed", details=details) from exc
            self._imports = imports
            self._topology = topology
            self._attention_backend = attention_backend
            self._supported_runner_types = supported_runner_types
            self._capabilities = capabilities
            self._llm = llm

    def capabilities(self) -> RuntimeCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        version = self._imports.version if self._imports is not None else None
        return default_vllm_capabilities(
            self.config.model,
            version,
            attention_backend=self._attention_backend,
            supported_runner_types=self._supported_runner_types,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )

    def topology(self) -> RuntimeTopology | None:
        return self._topology

    def list_models(self) -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "wllm",
                }
            ],
        }

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        self._ensure_loaded()
        prompt = self._render_chat(request)
        return self._generate_openai_response(request, [prompt], chat=True)

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        self._ensure_loaded()
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        return self._generate_openai_response(request, prompts, chat=False)

    def generate_extract(
        self,
        request: ExtractRequest,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
    ) -> TraceEnvelope:
        if request.n != 1:
            raise InvalidRequestError(
                "Extraction requests currently support exactly one completion. Set n=1.",
                code="unsupported_extraction_sample_count",
                param="n",
                details={"requested": request.n, "supported": 1},
            )
        request_id = f"extract_{uuid.uuid4().hex}"
        collector_registry = self._collector_registry
        if collector_registry is None:
            collector_registry = CollectorRegistry()
            self._collector_registry = collector_registry
        with collector_registry.scope(request_id):
            self._ensure_loaded()
            orchestrator = ExtractionOrchestrator(self.capabilities())
            topology = self.topology()
            orchestrator.preflight(request, limits, topology=topology)
            prompt = self._render_extract_prompt(request)
            sampling = self._sampling_params(request, force_logprobs=request.extract.logprobs is not None)
            started = time.perf_counter()
            outputs = self._llm.generate([prompt], sampling)
            generation_ms = (time.perf_counter() - started) * 1000.0
            output = outputs[0]
            completion = output.outputs[0]
            generation = self._generation_summary_from_vllm(output, completion, chat=request.messages is not None)
            prompt_ids = list(getattr(output, "prompt_token_ids", []) or [])
            generated_ids = list(getattr(completion, "token_ids", []) or [])
            token_ids = prompt_ids + generated_ids
            plan = self._compile_post_generation_plan(request, prompt_ids, generated_ids, limits, topology)
            capture_started = time.perf_counter()
            hidden_states, hidden_capture_site = self._extract_hidden_states_if_requested(
                request,
                token_ids,
                plan,
                topology,
                limits,
            )
            capture_ms = (time.perf_counter() - capture_started) * 1000.0 if request.extract.hidden_states else 0.0
            inputs = GeneratedTraceInputs(
                model=self.served_model_name,
                generation=generation,
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated_ids,
                decoded_tokens=self._decode_tokens(token_ids),
                prompt_logprobs=self._logprob_candidates_from_entries(getattr(output, "prompt_logprobs", None)),
                generated_logprobs=self._logprob_candidates(completion),
                hidden_states=hidden_states,
                hidden_state_capture_site=hidden_capture_site,
                generation_ms=generation_ms,
                capture_ms=capture_ms,
                topology=topology,
            )
            return orchestrator.build_trace(
                request,
                inputs,
                limits=limits,
                artifact_store=artifact_store,
                persist=persist,
                plan=plan,
            )

    def _compile_post_generation_plan(
        self,
        request: ExtractRequest,
        prompt_ids: list[int],
        generated_ids: list[int],
        limits: ResourceLimits,
        topology: RuntimeTopology | None,
    ) -> ExtractionPlan | None:
        if topology is None:
            return None
        return compile_extraction_plan(
            request.extract,
            num_layers=topology.num_layers,
            num_heads=topology.num_attention_heads,
            prompt_token_count=len(prompt_ids),
            generated_token_count=len(generated_ids),
            limits=limits,
        )

    def _extract_hidden_states_if_requested(
        self,
        request: ExtractRequest,
        token_ids: list[int],
        plan: ExtractionPlan | None,
        topology: RuntimeTopology | None,
        limits: ResourceLimits,
    ) -> tuple[dict[int, Any], str]:
        if not request.extract.hidden_states:
            return {}, "transformer_block_output"
        if topology is None or plan is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime topology so selectors can be resolved.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        requested_layers = sorted({layer for selection in plan.hidden_states for layer in selection["layers"]})
        if topology.hidden_size is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime hidden_size so raw capture resource limits can be enforced.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        if self.config.tensor_parallel_size != 1:
            raise UnsupportedExtractionError(
                "Scoped hidden-state capture is currently supported only for tensor_parallel_size=1.",
                code="hidden_state_parallelism_unavailable",
                param="extract.hidden_states",
                details={"tensor_parallel_size": self.config.tensor_parallel_size},
            )
        if self.config.gpu_memory_utilization > 0.5:
            raise UnsupportedExtractionError(
                "Scoped hidden-state capture requires gpu_memory_utilization <= 0.5 so the isolated pooling runner can initialize alongside the generation runner.",
                code="hidden_state_memory_configuration_unavailable",
                param="extract.hidden_states",
                details={
                    "gpu_memory_utilization": self.config.gpu_memory_utilization,
                    "maximum_supported_for_hidden_states": 0.5,
                },
            )
        estimated_bytes = _estimate_hidden_capture_bytes(
            layer_count=len(requested_layers),
            token_count=len(token_ids),
            topology=topology,
            dtype=self.config.dtype,
        )
        if estimated_bytes > limits.max_total_captured_tensor_bytes:
            raise ResourceLimitError(
                "Requested hidden-state capture exceeds the total captured tensor byte limit before token-position selection.",
                param="extract.hidden_states",
                details={
                    "estimated_raw_capture_bytes": estimated_bytes,
                    "limit": limits.max_total_captured_tensor_bytes,
                    "captured_layers": len(requested_layers),
                    "captured_tokens_per_layer": len(token_ids),
                    "hidden_size": topology.hidden_size,
                    "dtype": self.config.dtype,
                },
            )
        return self._pooling_hidden_states(token_ids, requested_layers), "transformer_block_output"

    def _pooling_hidden_states(self, token_ids: list[int], layers: list[int]) -> dict[int, Any]:
        with self._pooling_lock:
            pooling_llm = self._ensure_pooling_llm()
            return capture_pooling_hidden_states(pooling_llm, token_ids=token_ids, layers=layers)

    def _ensure_pooling_llm(self) -> Any:
        if self._pooling_llm is not None:
            return self._pooling_llm
        imports = self._imports or import_vllm()
        if "pooling" not in (self._supported_runner_types or ["pooling"]):
            raise UnsupportedExtractionError(
                "The loaded vLLM model configuration does not advertise a pooling runner.",
                code="hidden_states_unavailable",
                param="extract.hidden_states",
                details={"supported_runner_types": self._supported_runner_types},
            )
        kwargs = supported_kwargs(
            imports.LLM,
            {
                "model": self.config.model,
                "dtype": self.config.dtype,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "tokenizer": self.config.tokenizer,
                "trust_remote_code": self.config.trust_remote_code,
                **pooling_runner_kwargs(imports.version),
            },
        )
        env_overrides = pooling_runner_environment(imports.version)
        with _temporary_environment(env_overrides):
            try:
                pooling_llm = imports.LLM(**kwargs)
            except Exception as exc:
                raise UnsupportedExtractionError(
                    "Could not initialize an isolated vLLM pooling runner for hidden-state extraction.",
                    code="hidden_states_unavailable",
                    param="extract.hidden_states",
                    details={"exception": repr(exc), **pooling_runner_kwargs(imports.version), **env_overrides},
                ) from exc
        if not hasattr(pooling_llm, "encode"):
            raise UnsupportedExtractionError(
                "The active vLLM version does not expose LLM.encode for pooling hidden-state extraction.",
                code="hidden_states_unavailable",
                param="extract.hidden_states",
            )
        self._pooling_llm = pooling_llm
        return self._pooling_llm

    def _render_chat(self, request: ChatCompletionRequest) -> str:
        tokenizer = self._llm.get_tokenizer()
        messages = [message.model_dump() for message in request.messages]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _render_extract_prompt(self, request: ExtractRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        tokenizer = self._llm.get_tokenizer()
        messages = [message.model_dump() for message in request.messages or []]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _sampling_params(self, request: Any, *, force_logprobs: bool = False) -> Any:
        imports = self._imports or import_vllm()
        logprobs = None
        prompt_logprobs = None
        requested_logprobs = getattr(request, "logprobs", None)
        if force_logprobs:
            top_k = getattr(getattr(request, "extract", None), "logprobs", None)
            logprobs = top_k.top_k if top_k and top_k.top_k is not None else 1
            if top_k and top_k.include_prompt:
                prompt_logprobs = logprobs
        elif requested_logprobs is True:
            logprobs = 1
        elif isinstance(requested_logprobs, int) and not isinstance(requested_logprobs, bool):
            logprobs = requested_logprobs
        seed = request.seed if request.seed is not None else self.config.seed
        candidates = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop,
            "n": request.n,
            "seed": seed,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "logprobs": logprobs,
            "prompt_logprobs": prompt_logprobs,
        }
        self._reject_unsupported_sampling_params(imports.SamplingParams, request, candidates, force_logprobs=force_logprobs)
        kwargs = supported_kwargs(imports.SamplingParams, candidates)
        return imports.SamplingParams(**kwargs)

    def _reject_unsupported_sampling_params(
        self,
        sampling_params_cls: Any,
        request: Any,
        candidates: dict[str, Any],
        *,
        force_logprobs: bool,
    ) -> None:
        supported = supported_parameter_names(sampling_params_cls)
        if supported is None:
            return
        requested = set(getattr(request, "model_fields_set", set()))
        # max_tokens is always part of the serving contract because wllm supplies
        # a default even when the client omits it.
        required = {"max_tokens"}
        requested_logprobs = getattr(request, "logprobs", None)
        for name, value in candidates.items():
            if not force_logprobs and name == "logprobs" and (requested_logprobs is False or requested_logprobs is None):
                continue
            if name in {"logprobs", "prompt_logprobs"} and value is not None:
                required.add(name)
            elif name == "seed" and value is not None and self.config.seed is not None:
                required.add(name)
            elif name in requested:
                required.add(name)
        unsupported = sorted(name for name in required if name not in supported)
        if not unsupported:
            return
        first = unsupported[0]
        if first == "logprobs" and force_logprobs:
            raise UnsupportedExtractionError(
                "Token logprobs are not supported by the active vLLM SamplingParams.",
                code="token_logprobs_unavailable",
                param="extract.logprobs",
                details={"unsupported": unsupported, "supported": sorted(supported)},
            )
        elif first == "prompt_logprobs" and force_logprobs:
            raise UnsupportedExtractionError(
                "Prompt logprobs are not supported by the active vLLM SamplingParams.",
                code="prompt_logprobs_unavailable",
                param="extract.logprobs.include_prompt",
                details={"unsupported": unsupported, "supported": sorted(supported)},
            )
        else:
            param = first
        raise InvalidRequestError(
            f"Sampling parameter {first!r} is not supported by the active vLLM SamplingParams.",
            code="unsupported_sampling_parameter",
            param=param,
            details={"unsupported": unsupported, "supported": sorted(supported)},
        )

    def _generate_openai_response(self, request: Any, prompts: list[str], *, chat: bool) -> dict[str, Any]:
        sampling = self._sampling_params(request)
        created = int(time.time())
        outputs = self._llm.generate(prompts, sampling)
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for prompt_index, output in enumerate(outputs):
            prompt_tokens += len(getattr(output, "prompt_token_ids", []) or [])
            for choice_index, completion in enumerate(output.outputs):
                completion_tokens += len(getattr(completion, "token_ids", []) or [])
                index = prompt_index * request.n + choice_index
                if chat:
                    choice = {
                        "index": index,
                        "message": {"role": "assistant", "content": completion.text},
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                else:
                    choice = {
                        "index": index,
                        "text": completion.text,
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                logprobs = self._openai_logprobs(completion)
                if logprobs is not None:
                    choice["logprobs"] = logprobs
                choices.append(choice)
        return {
            "id": f"{'chatcmpl' if chat else 'cmpl'}_{uuid.uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": created,
            "model": self.served_model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _generation_summary_from_vllm(self, output: Any, completion: Any, *, chat: bool) -> dict[str, Any]:
        prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
        completion_tokens = len(getattr(completion, "token_ids", []) or [])
        choice = {
            "index": 0,
            "finish_reason": getattr(completion, "finish_reason", None),
        }
        if chat:
            choice["message"] = {"role": "assistant", "content": completion.text}
        else:
            choice["text"] = completion.text
        return {
            "id": f"{'chatcmpl' if chat else 'cmpl'}_{uuid.uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _decode_tokens(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        tokenizer = self._llm.get_tokenizer()
        decoded = []
        for token_id in token_ids:
            try:
                decoded.append(tokenizer.decode([token_id]))
            except Exception:
                decoded.append(f"<token:{token_id}>")
        return decoded

    def _logprob_candidates(self, completion: Any) -> list[list[LogprobCandidate]]:
        return self._logprob_candidates_from_entries(getattr(completion, "logprobs", None))

    def _logprob_candidates_from_entries(self, entries: Any) -> list[list[LogprobCandidate]]:
        rows = []
        for entry in entries or []:
            candidates: list[LogprobCandidate] = []
            if not entry:
                rows.append(candidates)
                continue
            for token_id, logprob_obj in entry.items():
                logprob = float(getattr(logprob_obj, "logprob", logprob_obj))
                candidates.append(LogprobCandidate(token_id=int(token_id), token=getattr(logprob_obj, "decoded_token", None), logprob=logprob))
            rows.append(candidates)
        return rows

    def _openai_logprobs(self, completion: Any) -> dict[str, Any] | None:
        entries = getattr(completion, "logprobs", None)
        if not entries:
            return None
        tokens = []
        token_logprobs = []
        top_logprobs = []
        for entry in entries:
            if not entry:
                tokens.append(None)
                token_logprobs.append(None)
                top_logprobs.append({})
                continue
            first_token_id, first = next(iter(entry.items()))
            tokens.append(getattr(first, "decoded_token", str(first_token_id)))
            token_logprobs.append(float(getattr(first, "logprob", first)))
            top_logprobs.append(
                {
                    str(token_id): float(getattr(logprob_obj, "logprob", logprob_obj))
                    for token_id, logprob_obj in entry.items()
                }
            )
        return {"tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top_logprobs}


class _temporary_environment:
    def __init__(self, overrides: dict[str, str]) -> None:
        self.overrides = overrides
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.overrides.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _estimate_hidden_capture_bytes(
    *,
    layer_count: int,
    token_count: int,
    topology: RuntimeTopology,
    dtype: str,
) -> int:
    return layer_count * token_count * int(topology.hidden_size or 0) * _dtype_byte_size(dtype)


def _dtype_byte_size(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in {"float16", "half", "bfloat16"}:
        return 2
    if normalized in {"float64", "double"}:
        return 8
    if normalized in {"int8", "uint8"}:
        return 1
    if normalized in {"float32", "float", "auto"}:
        return 4
    return 4
