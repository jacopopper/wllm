from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from artifacts.store import ArtifactStore
from extractors.collectors import CollectorRegistry
from extractors.hidden_states import hidden_states_unavailable
from extractors.planning import ExtractionPlan, ResourceLimits, RuntimeTopology, compile_extraction_plan
from extractors.selectors import SelectorValidationError, normalize_layer_selector
from runtime.capabilities import RuntimeCapabilities, default_vllm_capabilities
from runtime.orchestration import ExtractionOrchestrator, GeneratedTraceInputs, LogprobCandidate
from runtime.vllm_compat import (
    SUPPORTED_VLLM_VERSION,
    OnlineHiddenStateSelection,
    capture_online_hidden_states,
    capture_pooling_hidden_states,
    capture_transformers_replay_attentions,
    extract_attention_backend,
    extract_model_topology,
    extract_supported_runner_types,
    import_vllm,
    load_transformers_attention_replay,
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
    prewarm_hidden_states: bool = False
    enable_online_hidden_states: bool = False
    enable_attention_weights: bool = False


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
        self._attention_replay: Any | None = None
        self._prewarm_initializing = False
        self._load_lock = threading.RLock()
        self._generation_lock = threading.RLock()
        self._pooling_lock = threading.RLock()
        self._attention_replay_lock = threading.RLock()

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
                    "enforce_eager": True if self._uses_eager_generation_runner else None,
                    "enable_prefix_caching": False if self._uses_eager_generation_runner else None,
                },
            )
            env_overrides = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"} if self._uses_eager_generation_runner else {}
            try:
                with _temporary_environment(env_overrides):
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
                    online_hidden_states=self.config.enable_online_hidden_states,
                    attention_weights=self.config.enable_attention_weights,
                )
            except Exception as exc:
                details: dict[str, Any] = {"exception": repr(exc), "model": self.config.model}
                if env_overrides:
                    details.update(env_overrides)
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
            online_hidden_states=self.config.enable_online_hidden_states,
            attention_weights=self.config.enable_attention_weights,
        )

    def topology(self) -> RuntimeTopology | None:
        return self._topology

    def _require_llm(self) -> Any:
        if self._llm is None:
            raise AssertionError("vLLM runtime is not loaded")
        return self._llm

    @property
    def _uses_eager_generation_runner(self) -> bool:
        return self.config.enable_online_hidden_states or self._prewarm_initializing

    def prewarm_hidden_states(self) -> None:
        """Initialize the optional hidden-state extraction runner before serving requests."""

        self._prewarm_initializing = True
        try:
            self._ensure_loaded()
        finally:
            self._prewarm_initializing = False
        self._ensure_replay_hidden_state_runtime_available(self._topology)
        with self._pooling_lock:
            self._ensure_pooling_llm()

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
            llm = self._require_llm()
            orchestrator = ExtractionOrchestrator(self.capabilities())
            topology = self.topology()
            orchestrator.preflight(request, limits, topology=topology)
            prompt = self._render_extract_prompt(request)
            # For extraction we always want the chosen token's logprob available
            # (makes common white-box UQ baselines trivial). Full top-k only if requested.
            force_logprobs = True  # at minimum for chosen token prob
            if request.extract.logprobs is not None:
                force_logprobs = True
            sampling = self._sampling_params(request, force_logprobs=force_logprobs)
            capture_mode = self._hidden_state_capture_mode(request)
            online_capture = None
            online_plan: dict[str, ExtractionPlan | None] | None = None
            if capture_mode == "replay":
                self._ensure_replay_hidden_state_runtime_available(topology)
            started = time.perf_counter()
            with self._generation_lock:
                if capture_mode == "online":
                    requested_layers = self._resolve_hidden_state_layers(request, topology)
                    self._ensure_online_hidden_state_runtime_available(topology)
                    assert topology is not None
                    prompt_token_count = self._tokenized_prompt_length(prompt)
                    self._check_hidden_capture_size_for_shape(
                        request,
                        layer_count=len(requested_layers),
                        token_count=self._online_hidden_capture_token_count_upper_bound(
                            prompt,
                            request,
                            prompt_token_count=prompt_token_count,
                        ),
                        topology=topology,
                        limits=limits,
                    )
                    online_plan = {"plan": None}
                    online_capture = capture_online_hidden_states(
                        llm,
                        layers=requested_layers,
                        generate=lambda: self._generate_with_tqdm_disabled(llm, [prompt], sampling),
                        capture_max_position=self._online_hidden_capture_max_prefix_position(
                            request,
                            prompt_token_count=prompt_token_count,
                        ),
                        select_hidden_states=lambda outputs: self._online_hidden_state_selections_for_outputs(
                            request,
                            outputs,
                            limits=limits,
                            topology=topology,
                            plan_holder=online_plan,
                        ),
                    )
                    outputs = online_capture.output
                else:
                    outputs = self._generate_with_tqdm_disabled(llm, [prompt], sampling)
            generation_ms = (time.perf_counter() - started) * 1000.0
            output = outputs[0]
            completion = output.outputs[0]
            generation = self._generation_summary_from_vllm(output, completion, chat=request.messages is not None)
            prompt_ids = list(getattr(output, "prompt_token_ids", []) or [])
            generated_ids = list(getattr(completion, "token_ids", []) or [])
            token_ids = prompt_ids + generated_ids
            plan = (
                online_plan["plan"]
                if online_plan is not None and online_plan["plan"] is not None
                else self._compile_post_generation_plan(request, prompt_ids, generated_ids, limits, topology)
            )
            hidden_capture_mode = capture_mode or "replay"
            hidden_capture_phase = "replay"
            hidden_capture_metadata: dict[str, Any] = {}
            preselected_hidden_states: dict[str, Any] = {}
            attention_capture_metadata: dict[str, Any] = {}
            if capture_mode == "online":
                if online_capture is None:
                    raise AssertionError("online capture mode did not produce a capture result")
                hidden_states = online_capture.tensors
                preselected_hidden_states = online_capture.selected_tensors
                hidden_capture_site = online_capture.capture_site
                hidden_capture_phase = online_capture.capture_phase
                hidden_capture_metadata = online_capture.metadata
                capture_ms = online_capture.overhead_ms
                self._check_hidden_capture_size(
                    request,
                    token_ids,
                    plan,
                    topology,
                    limits,
                )
            else:
                capture_started = time.perf_counter()
                hidden_states, hidden_capture_site = self._extract_hidden_states_if_requested(
                    request,
                    token_ids,
                    plan,
                    topology,
                    limits,
                )
                capture_ms = (time.perf_counter() - capture_started) * 1000.0 if request.extract.hidden_states else 0.0
            attention_started = time.perf_counter()
            attentions, attention_capture_metadata = self._extract_attention_weights_if_requested(
                request,
                token_ids,
                plan,
                topology,
                limits,
            )
            if request.extract.attentions:
                capture_ms += (time.perf_counter() - attention_started) * 1000.0
            inputs = GeneratedTraceInputs(
                model=self.served_model_name,
                generation=generation,
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated_ids,
                decoded_tokens=self._decoded_tokens_for_request(request, token_ids),
                prompt_logprobs=self._logprob_candidates_from_entries(getattr(output, "prompt_logprobs", None)),
                generated_logprobs=self._logprob_candidates(completion),
                hidden_states=hidden_states,
                preselected_hidden_states=preselected_hidden_states,
                hidden_state_capture_site=hidden_capture_site,
                hidden_state_capture_mode=hidden_capture_mode,
                hidden_state_capture_phase=hidden_capture_phase,
                hidden_state_capture_metadata=hidden_capture_metadata,
                attentions=attentions,
                attention_capture_metadata=attention_capture_metadata,
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

    def _online_hidden_state_selections_for_outputs(
        self,
        request: ExtractRequest,
        outputs: list[Any],
        *,
        limits: ResourceLimits,
        topology: RuntimeTopology,
        plan_holder: dict[str, ExtractionPlan | None],
    ) -> list[OnlineHiddenStateSelection]:
        if not outputs:
            plan_holder["plan"] = None
            return []
        output = outputs[0]
        completions = list(getattr(output, "outputs", []) or [])
        if not completions:
            plan_holder["plan"] = None
            return []
        prompt_ids = list(getattr(output, "prompt_token_ids", []) or [])
        generated_ids = list(getattr(completions[0], "token_ids", []) or [])
        plan = self._compile_post_generation_plan(request, prompt_ids, generated_ids, limits, topology)
        plan_holder["plan"] = plan
        if plan is None:
            return []
        prompt_token_count = len(prompt_ids)
        return [
            OnlineHiddenStateSelection(
                name=f"hidden_states_{index}",
                layers=selection["layers"],
                positions=_source_positions_for_online_capture(
                    selection["positions"],
                    prompt_token_count=prompt_token_count,
                ),
                pool=selection["pool"],
            )
            for index, selection in enumerate(plan.hidden_states)
        ]

    def _generate_with_tqdm_disabled(self, llm: Any, prompts: list[str], sampling: Any) -> Any:
        with self._generation_lock:
            try:
                parameter_names = supported_parameter_names(llm.generate)
            except (TypeError, ValueError):
                parameter_names = None
            if parameter_names is None or "use_tqdm" in parameter_names:
                try:
                    return llm.generate(prompts, sampling, use_tqdm=False)
                except TypeError:
                    return llm.generate(prompts, sampling)
            return llm.generate(prompts, sampling)

    def _hidden_state_capture_mode(self, request: ExtractRequest) -> str | None:
        if not request.extract.hidden_states:
            return None
        modes = {hidden.capture_mode for hidden in request.extract.hidden_states}
        if len(modes) > 1:
            raise UnsupportedExtractionError(
                "A single extraction request cannot mix hidden-state capture modes yet.",
                code="mixed_hidden_state_capture_modes_unsupported",
                param="extract.hidden_states",
                details={"capture_modes": sorted(modes)},
            )
        return next(iter(modes))

    def _resolve_hidden_state_layers(self, request: ExtractRequest, topology: RuntimeTopology | None) -> list[int]:
        if topology is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime topology so selectors can be resolved.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        layers: set[int] = set()
        try:
            for index, hidden in enumerate(request.extract.hidden_states):
                for layer in normalize_layer_selector(hidden.layers, topology.num_layers):
                    layers.add(layer)
        except SelectorValidationError as exc:
            raise InvalidRequestError(str(exc), code="invalid_selector", param=exc.param) from exc
        return sorted(layers)

    def _ensure_online_hidden_state_runtime_available(self, topology: RuntimeTopology | None) -> None:
        if not self.config.enable_online_hidden_states:
            raise UnsupportedExtractionError(
                "Online hidden-state capture requires starting wllm with --enable-online-hidden-states.",
                code="online_hidden_states_disabled",
                param="extract.hidden_states.capture_mode",
            )
        if topology is None or topology.hidden_size is None:
            raise UnsupportedExtractionError(
                "Online hidden-state extraction requires runtime topology and hidden_size.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        version = self._imports.version if self._imports is not None else None
        if version != str(SUPPORTED_VLLM_VERSION):
            raise UnsupportedExtractionError(
                "Online hidden-state capture is validated only for vLLM 0.10.2.",
                code="online_hidden_states_unsupported_vllm_version",
                param="extract.hidden_states.capture_mode",
                details={"supported": str(SUPPORTED_VLLM_VERSION), "installed": version},
            )
        if self.config.tensor_parallel_size != 1:
            raise UnsupportedExtractionError(
                "Online hidden-state capture currently supports only tensor_parallel_size=1.",
                code="online_hidden_state_parallelism_unavailable",
                param="extract.hidden_states.capture_mode",
                details={"tensor_parallel_size": self.config.tensor_parallel_size},
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
        self._ensure_replay_hidden_state_runtime_available(topology)
        self._check_hidden_capture_size_for_shape(
            request,
            layer_count=len(requested_layers),
            token_count=len(token_ids),
            topology=topology,
            limits=limits,
        )
        return self._pooling_hidden_states(token_ids, requested_layers), "transformer_block_output"

    def _check_hidden_capture_size(
        self,
        request: ExtractRequest,
        token_ids: list[int],
        plan: ExtractionPlan | None,
        topology: RuntimeTopology | None,
        limits: ResourceLimits,
    ) -> None:
        if not request.extract.hidden_states:
            return
        if topology is None or plan is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime topology so selectors can be resolved.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        requested_layers = sorted({layer for selection in plan.hidden_states for layer in selection["layers"]})
        self._check_hidden_capture_size_for_shape(
            request,
            layer_count=len(requested_layers),
            token_count=len(token_ids),
            topology=topology,
            limits=limits,
        )

    def _check_hidden_capture_size_for_shape(
        self,
        request: ExtractRequest,
        *,
        layer_count: int,
        token_count: int,
        topology: RuntimeTopology,
        limits: ResourceLimits,
    ) -> None:
        if not request.extract.hidden_states:
            return
        estimated_bytes = _estimate_hidden_capture_bytes(
            layer_count=layer_count,
            token_count=token_count,
            topology=topology,
            dtype=self.config.dtype,
        )
        if estimated_bytes > limits.max_total_captured_tensor_bytes:
            raise ResourceLimitError(
                "Requested hidden-state capture exceeds the total "
                "captured tensor byte limit before token-position "
                "selection.",
                param="extract.hidden_states",
                details={
                    "estimated_raw_capture_bytes": estimated_bytes,
                    "limit": limits.max_total_captured_tensor_bytes,
                    "captured_layers": layer_count,
                    "captured_tokens_per_layer": token_count,
                    "hidden_size": topology.hidden_size,
                    "dtype": self.config.dtype,
                },
            )

    def _extract_attention_weights_if_requested(
        self,
        request: ExtractRequest,
        token_ids: list[int],
        plan: ExtractionPlan | None,
        topology: RuntimeTopology | None,
        limits: ResourceLimits,
    ) -> tuple[dict[int, Any], dict[str, Any]]:
        if not request.extract.attentions:
            return {}, {}
        self._ensure_attention_replay_runtime_available(topology)
        if topology is None or plan is None or topology.num_attention_heads is None:
            raise UnsupportedExtractionError(
                "Attention extraction requires runtime topology so selectors can be resolved.",
                code="attention_topology_unavailable",
                param="extract.attentions",
            )
        self._check_attention_capture_size(request, token_ids, plan, topology, limits)
        requested_layers = sorted({layer for selection in plan.attentions for layer in selection["layers"]})
        with self._attention_replay_lock:
            replay = self._ensure_attention_replay()
            captured = capture_transformers_replay_attentions(replay, token_ids=token_ids)
        missing_layers = [layer for layer in requested_layers if layer not in captured]
        if missing_layers:
            raise UnsupportedExtractionError(
                "The Transformers replay path did not return every requested attention layer.",
                code="attention_layer_unavailable",
                param="extract.attentions",
                details={"missing_layers": missing_layers, "captured_layers": sorted(captured)},
            )
        metadata = {
            "backend": "transformers_replay",
            "requested_layers": requested_layers,
            "replayed_token_count": len(token_ids),
            "model": self.config.model,
            "tokenizer": self.config.tokenizer or self.config.model,
            "use_cache": False,
            "output_attentions": True,
            "replay_device": getattr(replay, "device", None),
        }
        return {layer: captured[layer] for layer in requested_layers}, metadata

    def _ensure_attention_replay_runtime_available(self, topology: RuntimeTopology | None) -> None:
        if not self.config.enable_attention_weights:
            from extractors.attentions import attentions_unavailable

            attentions_unavailable(details={"capability": self.capabilities().attentions.model_dump(mode="json")})
        if topology is None or topology.num_attention_heads is None:
            raise UnsupportedExtractionError(
                "Attention extraction requires runtime topology with num_attention_heads.",
                code="attention_topology_unavailable",
                param="extract.attentions",
            )

    def _check_attention_capture_size(
        self,
        request: ExtractRequest,
        token_ids: list[int],
        plan: ExtractionPlan,
        topology: RuntimeTopology,
        limits: ResourceLimits,
    ) -> None:
        if not request.extract.attentions:
            return
        if topology.num_attention_heads is None:
            raise UnsupportedExtractionError(
                "Attention extraction requires runtime num_attention_heads so resource limits can be enforced.",
                code="attention_topology_unavailable",
                param="extract.attentions",
            )
        dtype_bytes = _dtype_byte_size(self.config.dtype)
        replay_bytes = _estimate_attention_capture_bytes(
            layer_count=topology.num_layers,
            head_count=topology.num_attention_heads,
            token_count=len(token_ids),
            dtype_bytes=dtype_bytes,
        )
        if replay_bytes > limits.max_total_captured_tensor_bytes:
            raise ResourceLimitError(
                "Requested attention replay exceeds the total captured tensor byte limit before selection.",
                param="extract.attentions",
                details={
                    "estimated_raw_capture_bytes": replay_bytes,
                    "limit": limits.max_total_captured_tensor_bytes,
                    "captured_layers": topology.num_layers,
                    "captured_heads_per_layer": topology.num_attention_heads,
                    "captured_tokens": len(token_ids),
                    "dtype": self.config.dtype,
                },
            )
        selected_bytes = 0
        for selection in plan.attentions:
            head_count = (
                topology.num_attention_heads
                if selection["heads"] == "all"
                else len(selection["heads"])
            )
            max_key_count = max(
                (len(selection["key_positions"].get(query, [])) for query in selection["query_positions"]),
                default=0,
            )
            selected_bytes += _estimate_attention_capture_bytes(
                layer_count=len(selection["layers"]),
                head_count=head_count,
                token_count=1,
                dtype_bytes=dtype_bytes,
                query_count=len(selection["query_positions"]),
                key_count=max_key_count,
            )
        if selected_bytes > limits.max_total_captured_tensor_bytes:
            raise ResourceLimitError(
                "Requested selected attention tensors exceed the total captured tensor byte limit.",
                param="extract.attentions",
                details={"estimated_selected_bytes": selected_bytes, "limit": limits.max_total_captured_tensor_bytes},
            )

    def _ensure_attention_replay(self) -> Any:
        if self._attention_replay is not None:
            return self._attention_replay
        self._attention_replay = load_transformers_attention_replay(
            model=self.config.model,
            tokenizer=self.config.tokenizer,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        return self._attention_replay

    def _online_hidden_capture_token_count_upper_bound(
        self,
        prompt: str,
        request: ExtractRequest,
        *,
        prompt_token_count: int | None = None,
    ) -> int:
        prompt_count = prompt_token_count if prompt_token_count is not None else self._tokenized_prompt_length(prompt)
        if prompt_count is not None:
            return prompt_count + int(request.max_tokens)
        if self.config.max_model_len is not None:
            return self.config.max_model_len
        return len(prompt.encode("utf-8")) + int(request.max_tokens)

    def _online_hidden_capture_max_prefix_position(
        self,
        request: ExtractRequest,
        *,
        prompt_token_count: int | None,
    ) -> int | None:
        if prompt_token_count is None:
            return None
        max_position = -1
        for hidden in request.extract.hidden_states:
            prefix_position = _prompt_prefix_position(hidden.positions, prompt_token_count)
            if prefix_position is None:
                return None
            max_position = max(max_position, prefix_position)
        return max_position if max_position >= 0 else None

    def _tokenized_prompt_length(self, prompt: str) -> int | None:
        tokenizer = self._require_llm().get_tokenizer()
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            lengths = []
            try:
                lengths.append(len(encode(prompt, add_special_tokens=False)))
            except TypeError:
                pass
            except Exception:
                pass
            try:
                lengths.append(len(encode(prompt, add_special_tokens=True)))
            except TypeError:
                try:
                    lengths.append(len(encode(prompt)))
                except Exception:
                    pass
            except Exception:
                pass
            if lengths:
                return max(lengths)
        tokenize = getattr(tokenizer, "__call__", None)
        if callable(tokenize):
            try:
                encoded = tokenize(prompt, add_special_tokens=True)
            except TypeError:
                try:
                    encoded = tokenize(prompt)
                except Exception:
                    return None
            except Exception:
                return None
            input_ids = getattr(encoded, "input_ids", None)
            if input_ids is None and isinstance(encoded, dict):
                input_ids = encoded.get("input_ids")
            if input_ids is not None:
                return len(input_ids)
        return None

    def _ensure_replay_hidden_state_runtime_available(self, topology: RuntimeTopology | None) -> None:
        capability = self._replay_hidden_state_capability()
        if capability.state == "unsupported":
            hidden_states_unavailable(details={"capability": capability.model_dump(mode="json")})
        self._ensure_hidden_state_topology_available(topology)

    def _ensure_hidden_state_topology_available(self, topology: RuntimeTopology | None) -> None:
        if topology is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime topology so selectors can be resolved.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        if topology.hidden_size is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime hidden_size so raw capture resource limits can be enforced.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )

    def _replay_hidden_state_capability(self) -> Any:
        version = self._imports.version if self._imports is not None else None
        return default_vllm_capabilities(
            self.config.model,
            version,
            attention_backend=self._attention_backend,
            supported_runner_types=self._supported_runner_types,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            online_hidden_states=False,
        ).hidden_states

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
        tokenizer = self._require_llm().get_tokenizer()
        messages = [message.model_dump() for message in request.messages]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _render_extract_prompt(self, request: ExtractRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        tokenizer = self._require_llm().get_tokenizer()
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
        self._reject_unsupported_sampling_params(
            imports.SamplingParams, request, candidates, force_logprobs=force_logprobs)
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
            if not force_logprobs and name == "logprobs" and (
                    requested_logprobs is False or requested_logprobs is None):
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
        outputs = self._generate_with_tqdm_disabled(self._require_llm(), prompts, sampling)
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
        tokenizer = self._require_llm().get_tokenizer()
        batch_decode = getattr(tokenizer, "batch_decode", None)
        if callable(batch_decode):
            try:
                return list(
                    batch_decode(
                        [[token_id] for token_id in token_ids],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
            except TypeError:
                try:
                    return list(batch_decode([[token_id] for token_id in token_ids]))
                except Exception:
                    pass
            except Exception:
                pass
        decoded = []
        for token_id in token_ids:
            try:
                decoded.append(tokenizer.decode([token_id]))
            except Exception:
                decoded.append(f"<token:{token_id}>")
        return decoded

    def _decoded_tokens_for_request(self, request: ExtractRequest, token_ids: list[int]) -> list[str]:
        if not request.extract.tokens and request.extract.logprobs is None:
            return []
        return self._decode_tokens(token_ids)

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
                candidates.append(
                    LogprobCandidate(
                        token_id=int(token_id),
                        token=getattr(
                            logprob_obj,
                            "decoded_token",
                            None),
                        logprob=logprob))
            rows.append(candidates)
        return rows

    def _openai_logprobs(self, completion: Any) -> dict[str, Any] | None:
        entries = getattr(completion, "logprobs", None)
        if not entries:
            return None
        tokens: list[str | None] = []
        token_logprobs: list[float | None] = []
        top_logprobs: list[dict[str, float]] = []
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


def _source_positions_for_online_capture(positions: list[int], *, prompt_token_count: int) -> list[int]:
    return [position if position < prompt_token_count else position - 1 for position in positions]


def _estimate_hidden_capture_bytes(
    *,
    layer_count: int,
    token_count: int,
    topology: RuntimeTopology,
    dtype: str,
) -> int:
    return layer_count * token_count * int(topology.hidden_size or 0) * _dtype_byte_size(dtype)


def _estimate_attention_capture_bytes(
    *,
    layer_count: int,
    head_count: int,
    token_count: int,
    dtype_bytes: int,
    query_count: int | None = None,
    key_count: int | None = None,
) -> int:
    queries = token_count if query_count is None else query_count
    keys = token_count if key_count is None else key_count
    return layer_count * head_count * queries * keys * dtype_bytes


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


def _prompt_prefix_position(selector: Any, prompt_token_count: int) -> int | None:
    if selector == "prompt":
        return prompt_token_count - 1 if prompt_token_count > 0 else None
    if isinstance(selector, int) and not isinstance(selector, bool):
        return selector if 0 <= selector < prompt_token_count else None
    if isinstance(selector, list):
        if not selector:
            return None
        positions = []
        for item in selector:
            if not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= prompt_token_count:
                return None
            positions.append(item)
        return max(positions)
    return None
