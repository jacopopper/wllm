from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from artifacts.store import ArtifactStore
from extractors.attentions import attentions_unavailable
from extractors.hidden_states import hidden_states_unavailable
from extractors.planning import (
    ExtractionPlan,
    ResourceLimits,
    RuntimeTopology,
    compile_extraction_plan,
    validate_pre_generation_selectors,
)
from extractors.token_data import approximate_entropy_from_top_logprobs
from runtime.capabilities import RuntimeCapabilities
from schemas.extraction import ExtractRequest
from schemas.traces import TensorRecord, TokenTrace, TraceData, TraceEnvelope, TraceMetadata
from server.errors import InvalidRequestError, ResourceLimitError, UnsupportedExtractionError
from tracing.context import active_trace_id


@dataclass(frozen=True)
class LogprobCandidate:
    token_id: int
    logprob: float
    token: str | None = None


@dataclass(frozen=True)
class GeneratedTraceInputs:
    model: str
    generation: dict[str, Any]
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    decoded_tokens: list[str] = field(default_factory=list)
    prompt_logprobs: list[list[LogprobCandidate]] = field(default_factory=list)
    generated_logprobs: list[list[LogprobCandidate]] = field(default_factory=list)
    hidden_states: dict[int, Any] = field(default_factory=dict)
    hidden_state_capture_site: str = "transformer_block_output"
    hidden_state_capture_mode: str = "replay"
    hidden_state_capture_phase: str = "replay"
    hidden_state_capture_metadata: dict[str, Any] = field(default_factory=dict)
    generation_ms: float = 0.0
    capture_ms: float = 0.0
    topology: RuntimeTopology | None = None


@dataclass(frozen=True)
class HiddenStateTensor:
    name: str
    tensor: Any
    layers: list[int]
    positions: list[int]
    pool: str | None
    capture_site: str
    capture_mode: str
    capture_phase: str
    position_semantics: dict[str, Any]
    capture_metadata: dict[str, Any]


class ExtractionOrchestrator:
    """Coordinates request validation, trace construction, and artifact persistence.

    This layer intentionally uses Python plus NumPy/PyTorch-compatible array layouts.
    It does not sit in the vLLM token-generation hot path.
    """

    def __init__(self, capabilities: RuntimeCapabilities) -> None:
        self.capabilities = capabilities

    def preflight(
        self,
        request: ExtractRequest,
        limits: ResourceLimits,
        *,
        topology: RuntimeTopology | None = None,
    ) -> None:
        spec = request.extract
        if spec.logprobs and spec.logprobs.top_k is not None and spec.logprobs.top_k > limits.max_top_k:
            raise ResourceLimitError(
                f"Requested top_k={spec.logprobs.top_k} exceeds max_top_k={limits.max_top_k}.",
                param="extract.logprobs.top_k",
                details={"requested": spec.logprobs.top_k, "limit": limits.max_top_k},
            )
        if topology is not None:
            validate_pre_generation_selectors(spec, topology=topology, limits=limits)
        if spec.logprobs and spec.logprobs.raw_logits:
            raise UnsupportedExtractionError(
                "Raw logits are not exposed by the public vLLM generation output.",
                code="raw_logits_unavailable",
                param="extract.logprobs.raw_logits",
            )
        if spec.logprobs and spec.logprobs.entropy and not spec.logprobs.allow_approximate_entropy:
            raise UnsupportedExtractionError(
                "Exact entropy requires the complete token distribution, which the active runtime does not expose.",
                code="exact_entropy_unavailable",
                param="extract.logprobs.entropy",
            )
        if spec.artifacts and "logprobs" in spec.artifacts.include and spec.logprobs is None:
            raise InvalidRequestError(
                "Logprob artifacts require extract.logprobs so the runtime requests logprobs from vLLM.",
                code="artifact_dependency_missing",
                param="extract.artifacts.include",
                details={"missing": "extract.logprobs", "artifact": "logprobs"},
            )
        replay_hidden_states = [hidden for hidden in spec.hidden_states if hidden.capture_mode == "replay"]
        if replay_hidden_states and self.capabilities.hidden_states.state == "unsupported":
            hidden_states_unavailable(
                details={"capability": self.capabilities.hidden_states.model_dump(mode="json")}
            )
        if spec.attentions:
            attentions_unavailable(
                details={"capability": self.capabilities.attentions.model_dump(mode="json")}
            )
        if spec.artifacts and "hidden_states" in spec.artifacts.include and not spec.hidden_states:
            raise InvalidRequestError(
                "Hidden-state artifacts require extract.hidden_states so the runtime knows what to capture.",
                code="artifact_dependency_missing",
                param="extract.artifacts.include",
                details={"missing": "extract.hidden_states", "artifact": "hidden_states"},
            )
        if (
            spec.artifacts
            and "hidden_states" in spec.artifacts.include
            and replay_hidden_states
            and self.capabilities.hidden_states.state == "unsupported"
        ):
            hidden_states_unavailable(
                details={"capability": self.capabilities.hidden_states.model_dump(mode="json")}
            )
        if spec.artifacts and "attentions" in spec.artifacts.include:
            raise UnsupportedExtractionError(
                "Requested artifact contents require unsupported tensor extraction.",
                code="artifact_contents_unavailable",
                param="extract.artifacts.include",
            )

    def build_trace(
        self,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
        plan: ExtractionPlan | None = None,
    ) -> TraceEnvelope:
        started = time.perf_counter()
        trace_id = f"trace_{uuid.uuid4().hex}"
        context_token = active_trace_id.set(trace_id)
        try:
            return self._build_trace_with_context(
                trace_id,
                request,
                inputs,
                limits=limits,
                artifact_store=artifact_store,
                persist=persist,
                started=started,
                plan=plan,
            )
        finally:
            active_trace_id.reset(context_token)

    def _build_trace_with_context(
        self,
        trace_id: str,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        *,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
        started: float,
        plan: ExtractionPlan | None,
    ) -> TraceEnvelope:
        prompt_count = len(inputs.prompt_token_ids)
        token_ids = inputs.prompt_token_ids + inputs.generated_token_ids
        if plan is None and inputs.topology is not None:
            plan = compile_extraction_plan(
                request.extract,
                num_layers=inputs.topology.num_layers,
                num_heads=inputs.topology.num_attention_heads,
                prompt_token_count=prompt_count,
                generated_token_count=len(inputs.generated_token_ids),
                limits=limits,
            )
        hidden_tensors = self._hidden_state_tensors(request, inputs, plan)
        self._check_inline_bytes(request, inputs, token_ids, limits, hidden_tensors)
        trace = TraceEnvelope(
            id=trace_id,
            created=int(time.time()),
            model=inputs.model,
            generation=inputs.generation,
            trace=TraceData(
                tokens=TokenTrace(
                    token_ids=token_ids if request.extract.tokens else [],
                    tokens=inputs.decoded_tokens if request.extract.tokens else [],
                ),
                spans={"prompt": (0, prompt_count), "generated": (prompt_count, len(token_ids))},
                logprobs=self._inline_logprobs(request, inputs),
            ),
            metadata=TraceMetadata(
                sampling={
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "n": request.n,
                    "seed": request.seed,
                },
                capabilities=self.capabilities.as_metadata(),
                resolved_selectors=plan.resolved_selectors if plan is not None else {},
                capture=self._capture_metadata(request, inputs, plan),
            ),
        )
        trace.metadata.timing_ms.generation = inputs.generation_ms
        trace.metadata.timing_ms.capture = inputs.capture_ms
        trace.metadata.timing_ms.postprocess = (time.perf_counter() - started) * 1000.0
        serialization_started = time.perf_counter()
        trace.artifacts.extend(
            self._write_requested_artifacts(
                request,
                inputs,
                trace.id,
                limits,
                artifact_store,
                persist,
                hidden_tensors=hidden_tensors,
            )
        )
        hidden_artifact_id = self._artifact_id_for_hidden_states(request, trace.artifacts)
        trace.trace.hidden_states = self._hidden_state_records(request, hidden_tensors, artifact_id=hidden_artifact_id)
        if persist:
            trace.trace_manifest = artifact_store.put_trace_bundle(
                trace_id=trace.id,
                schema_version=trace.schema_version,
                payload=trace.model_dump(mode="json", exclude_none=True),
            )
            if trace.trace_manifest.byte_size > limits.max_artifact_bytes:
                artifact_store.delete_manifest_path(trace.trace_manifest.path)
                raise ResourceLimitError(
                    "Serialized trace bundle exceeds the artifact byte limit.",
                    param="trace",
                    details={"trace_bundle_bytes": trace.trace_manifest.byte_size, "limit": limits.max_artifact_bytes},
                )
        trace.metadata.timing_ms.serialization = (time.perf_counter() - serialization_started) * 1000.0
        trace.metadata.timing_ms.extraction_overhead = (
            trace.metadata.timing_ms.capture
            + trace.metadata.timing_ms.postprocess
            + trace.metadata.timing_ms.serialization
        )
        trace.metadata.timing_ms.total = inputs.generation_ms + trace.metadata.timing_ms.extraction_overhead
        return trace

    def _inline_logprobs(self, request: ExtractRequest, inputs: GeneratedTraceInputs) -> dict[str, Any]:
        spec = request.extract.logprobs
        if spec is None:
            return {}
        top_k = spec.top_k or 1
        logprobs = {
            "generated": self._inline_logprob_rows(
                inputs.generated_logprobs,
                selected_token_ids=inputs.generated_token_ids,
                decoded_tokens=inputs.decoded_tokens[len(inputs.prompt_token_ids):],
                top_k=top_k,
                include_entropy=spec.entropy and spec.allow_approximate_entropy,
            )
        }
        if spec.include_prompt:
            logprobs["prompt"] = self._inline_logprob_rows(
                inputs.prompt_logprobs,
                selected_token_ids=inputs.prompt_token_ids,
                decoded_tokens=inputs.decoded_tokens[: len(inputs.prompt_token_ids)],
                top_k=top_k,
                include_entropy=spec.entropy and spec.allow_approximate_entropy,
            )
        return logprobs

    def _inline_logprob_rows(
        self,
        rows: list[list[LogprobCandidate]],
        *,
        selected_token_ids: list[int],
        decoded_tokens: list[str],
        top_k: int,
        include_entropy: bool,
    ) -> list[dict[str, Any]]:
        per_token = []
        for index, row in enumerate(rows):
            selected_token_id = selected_token_ids[index] if index < len(selected_token_ids) else None
            selected = self._selected_logprob(row, selected_token_id)
            selected_token = selected.token if selected and selected.token is not None else (
                decoded_tokens[index] if index < len(decoded_tokens) else None
            )
            candidates = [
                {"token_id": candidate.token_id, "token": candidate.token, "logprob": candidate.logprob}
                for candidate in row[:top_k]
            ]
            item: dict[str, Any] = {
                "token_id": selected_token_id,
                "token": selected_token,
                "logprob": selected.logprob if selected is not None else None,
                "top_logprobs": candidates,
            }
            if include_entropy:
                entropy = approximate_entropy_from_top_logprobs(
                    np.asarray([candidate["logprob"] for candidate in candidates], dtype=np.float64)
                )
                item["entropy"] = {"value": entropy, "approximation": "renormalized_top_k"}
            per_token.append(item)
        return per_token

    @staticmethod
    def _selected_logprob(row: list[LogprobCandidate], token_id: int | None) -> LogprobCandidate | None:
        if token_id is None:
            return None
        for candidate in row:
            if candidate.token_id == token_id:
                return candidate
        return None

    def _write_requested_artifacts(
        self,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        trace_id: str,
        limits: ResourceLimits,
        artifact_store: ArtifactStore,
        persist: bool,
        hidden_tensors: list[HiddenStateTensor] | None = None,
    ) -> list[Any]:
        hidden_tensors = hidden_tensors or []
        artifact_request = request.extract.artifacts
        if artifact_request is None:
            return []
        include = set(artifact_request.include)
        if persist and not include:
            include = {"tokens", "logprobs"} if request.extract.logprobs else {"tokens"}
        tensors: dict[str, Any] = {}
        if "tokens" in include:
            all_token_ids = inputs.prompt_token_ids + inputs.generated_token_ids
            tensors["token_ids"] = np.asarray(all_token_ids, dtype=np.int64)
        if "logprobs" in include:
            top_k = request.extract.logprobs.top_k if request.extract.logprobs else None
            generated_token_ids, generated_logprobs = self._logprob_arrays(inputs, top_k=top_k)
            tensors["generated_logprob_token_ids"] = generated_token_ids
            tensors["generated_logprobs"] = generated_logprobs
            if request.extract.logprobs and request.extract.logprobs.include_prompt:
                prompt_token_ids, prompt_logprobs = self._logprob_arrays(inputs, top_k=top_k, prompt=True)
                tensors["prompt_logprob_token_ids"] = prompt_token_ids
                tensors["prompt_logprobs"] = prompt_logprobs
        if "hidden_states" in include:
            for hidden in hidden_tensors:
                tensors[hidden.name] = hidden.tensor
        total_bytes = sum(_tensor_artifact_nbytes(tensor, artifact_request.format) for tensor in tensors.values())
        if total_bytes > limits.max_total_captured_tensor_bytes:
            raise ResourceLimitError(
                "Requested artifact tensors exceed the total captured tensor byte limit.",
                param="extract.artifacts.include",
                details={"requested_bytes": total_bytes, "limit": limits.max_total_captured_tensor_bytes},
            )
        if total_bytes > limits.max_artifact_bytes:
            raise ResourceLimitError(
                "Requested artifact tensors exceed the artifact byte limit.",
                param="extract.artifacts.include",
                details={"requested_bytes": total_bytes, "limit": limits.max_artifact_bytes},
            )
        if not tensors:
            return []
        manifest = artifact_store.put(
            trace_id=trace_id,
            tensors=tensors,
            format=artifact_request.format,
            compression=artifact_request.compression,
        )
        if manifest.byte_size > limits.max_artifact_bytes:
            artifact_store.delete_manifest_path(manifest.path)
            raise ResourceLimitError(
                "Serialized artifact exceeds the artifact byte limit.",
                param="extract.artifacts",
                details={"artifact_bytes": manifest.byte_size, "limit": limits.max_artifact_bytes},
            )
        return [manifest]

    def _check_inline_bytes(
        self,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        token_ids: list[int],
        limits: ResourceLimits,
        hidden_tensors: list[HiddenStateTensor] | None = None,
    ) -> None:
        inline_bytes = 0
        if request.extract.tokens:
            inline_bytes += len(token_ids) * np.dtype(np.int64).itemsize
        spec = request.extract.logprobs
        if spec is not None:
            top_k = spec.top_k or 1
            rows = len(inputs.generated_logprobs)
            width = max((min(len(row), top_k) for row in inputs.generated_logprobs), default=0)
            inline_bytes += rows * width * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
            inline_bytes += rows * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
            if spec.include_prompt:
                rows = len(inputs.prompt_logprobs)
                width = max((min(len(row), top_k) for row in inputs.prompt_logprobs), default=0)
                inline_bytes += rows * width * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
                inline_bytes += rows * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
            if spec.entropy and spec.allow_approximate_entropy:
                entropy_rows = len(inputs.generated_logprobs)
                if spec.include_prompt:
                    entropy_rows += len(inputs.prompt_logprobs)
                inline_bytes += entropy_rows * np.dtype(np.float64).itemsize
        inline_hidden_bytes = 0
        if not self._hidden_states_artifact_requested(request):
            inline_hidden_bytes = sum(_tensor_storage_nbytes(hidden.tensor) for hidden in hidden_tensors or [])
            inline_bytes += inline_hidden_bytes
            if inline_hidden_bytes > limits.max_total_captured_tensor_bytes:
                raise ResourceLimitError(
                    "Requested hidden-state tensors exceed the total captured tensor byte limit.",
                    param="extract.hidden_states",
                    details={"requested_bytes": inline_hidden_bytes, "limit": limits.max_total_captured_tensor_bytes},
                )
        if inline_bytes > limits.max_inline_tensor_bytes:
            raise ResourceLimitError(
                "Requested inline extraction payload exceeds the inline tensor byte limit.",
                param="extract",
                details={
                    "requested_inline_bytes": inline_bytes,
                    "limit": limits.max_inline_tensor_bytes,
                    "hint": "Reduce max_tokens/top_k or request bounded tensor artifacts.",
                },
            )

    def _logprob_arrays(
        self,
        inputs: GeneratedTraceInputs,
        *,
        top_k: int | None = None,
        prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = inputs.prompt_logprobs if prompt else inputs.generated_logprobs
        width = max((min(len(row), top_k) if top_k is not None else len(row) for row in rows), default=0)
        token_ids = np.full((len(rows), width), -1, dtype=np.int64)
        logprobs = np.full((len(rows), width), np.nan, dtype=np.float32)
        if not rows or width == 0:
            return token_ids, logprobs
        for row_index, row in enumerate(rows):
            row_width = min(len(row), width)
            if row_width == 0:
                continue
            capped = row[:row_width]
            token_ids[row_index, :row_width] = np.asarray([candidate.token_id for candidate in capped], dtype=np.int64)
            logprobs[row_index, :row_width] = np.asarray([candidate.logprob for candidate in capped], dtype=np.float32)
        return token_ids, logprobs

    def _hidden_state_tensors(
        self,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        plan: ExtractionPlan | None,
    ) -> list[HiddenStateTensor]:
        if not request.extract.hidden_states:
            return []
        if plan is None:
            raise UnsupportedExtractionError(
                "Hidden-state extraction requires runtime topology so selectors can be resolved.",
                code="hidden_state_topology_unavailable",
                param="extract.hidden_states",
            )
        if not inputs.hidden_states:
            raise UnsupportedExtractionError(
                "Hidden-state extraction was requested, but the active vLLM path did not capture hidden-state tensors.",
                code="hidden_states_unavailable",
                param="extract.hidden_states",
                details={"capability": self.capabilities.hidden_states.model_dump(mode="json")},
            )
        hidden_tensors = []
        for index, selection in enumerate(plan.hidden_states):
            capture_mode = str(selection.get("capture_mode", inputs.hidden_state_capture_mode))
            capture_phase = _capture_phase_for_positions(
                selection["positions"],
                prompt_token_count=len(inputs.prompt_token_ids),
                capture_mode=capture_mode,
                default_phase=inputs.hidden_state_capture_phase,
            )
            source_positions = _source_positions_for_capture(
                selection["positions"],
                prompt_token_count=len(inputs.prompt_token_ids),
                capture_mode=capture_mode,
            )
            position_semantics = _hidden_state_position_semantics(
                prompt_token_count=len(inputs.prompt_token_ids),
                generated_token_count=len(inputs.generated_token_ids),
                selected_positions=selection["positions"],
                source_positions=source_positions,
                capture_mode=capture_mode,
                capture_phase=capture_phase,
            )
            tensor = self._select_hidden_state_tensor(
                inputs.hidden_states,
                layers=selection["layers"],
                positions=source_positions,
                pool=selection["pool"],
                param=f"extract.hidden_states[{index}]",
                token_context={
                    "prompt_token_count": len(inputs.prompt_token_ids),
                    "generated_token_count": len(inputs.generated_token_ids),
                    "total_token_count": len(inputs.prompt_token_ids) + len(inputs.generated_token_ids),
                    "requested_positions": selection["positions"],
                    "source_positions": source_positions,
                    "capture_mode": capture_mode,
                },
            )
            hidden_tensors.append(
                HiddenStateTensor(
                    name=f"hidden_states_{index}",
                    tensor=tensor,
                    layers=selection["layers"],
                    positions=selection["positions"],
                    pool=selection["pool"],
                    capture_site=inputs.hidden_state_capture_site,
                    capture_mode=capture_mode,
                    capture_phase=capture_phase,
                    position_semantics=position_semantics,
                    capture_metadata=inputs.hidden_state_capture_metadata,
                )
            )
        return hidden_tensors

    def _select_hidden_state_tensor(
        self,
        captured: dict[int, Any],
        *,
        layers: list[int],
        positions: list[int],
        pool: str | None,
        param: str,
        token_context: dict[str, Any] | None = None,
    ) -> Any:
        selected = []
        for layer in layers:
            source = captured.get(layer)
            if source is None:
                raise UnsupportedExtractionError(
                    f"Hidden states for layer {layer} were not captured by the active runtime path.",
                    code="hidden_state_layer_unavailable",
                    param=param,
                    details={"layer": layer, "captured_layers": sorted(captured)},
                )
            selected.append(_select_positions(source, positions, pool=pool, param=param, token_context=token_context))
        return _stack_tensors(selected, param=param)

    def _hidden_state_records(
        self,
        request: ExtractRequest,
        hidden_tensors: list[HiddenStateTensor],
        *,
        artifact_id: str | None,
    ) -> list[Any]:
        inline = not self._hidden_states_artifact_requested(request)
        records = []
        for hidden in hidden_tensors:
            capture_dtype = _tensor_dtype(hidden.tensor)
            storage_dtype = _tensor_storage_dtype(hidden.tensor)
            records.append(
                TensorRecord(
                    name=hidden.name,
                    shape=_tensor_shape(hidden.tensor),
                    dtype=capture_dtype,
                    capture_dtype=capture_dtype,
                    storage_dtype=storage_dtype,
                    device=_tensor_device(hidden.tensor),
                    layers=hidden.layers,
                    positions=hidden.positions,
                    capture_site=hidden.capture_site,
                    capture_mode=hidden.capture_mode,  # type: ignore[arg-type]
                    capture_phase=hidden.capture_phase,
                    position_semantics=hidden.position_semantics,
                    capture_metadata=hidden.capture_metadata,
                    data=_tensor_to_jsonable(hidden.tensor) if inline else None,
                    artifact_id=None if inline else artifact_id,
                    byte_size=_tensor_storage_nbytes(hidden.tensor),
                )
            )
        return records

    def _capture_metadata(
        self,
        request: ExtractRequest,
        inputs: GeneratedTraceInputs,
        plan: ExtractionPlan | None,
    ) -> dict[str, Any]:
        if not request.extract.hidden_states:
            return {}
        hidden_selectors = plan.hidden_states if plan is not None else []
        capture_modes = sorted({str(item.get("capture_mode", inputs.hidden_state_capture_mode))
                               for item in hidden_selectors})
        return {
            "hidden_states": {
                "capture_modes": capture_modes or [inputs.hidden_state_capture_mode],
                "capture_site": inputs.hidden_state_capture_site,
                "capture_phase": inputs.hidden_state_capture_phase,
                "position_semantics": _hidden_state_position_semantics(
                    prompt_token_count=len(inputs.prompt_token_ids),
                    generated_token_count=len(inputs.generated_token_ids),
                    selected_positions=[],
                    source_positions=[],
                    capture_mode=inputs.hidden_state_capture_mode,
                    capture_phase=inputs.hidden_state_capture_phase,
                ),
                "capture_metadata": inputs.hidden_state_capture_metadata,
            }
        }

    def _hidden_states_artifact_requested(self, request: ExtractRequest) -> bool:
        artifacts = request.extract.artifacts
        return bool(artifacts and "hidden_states" in artifacts.include)

    def _artifact_id_for_hidden_states(self, request: ExtractRequest, artifacts: list[Any]) -> str | None:
        if not self._hidden_states_artifact_requested(request):
            return None
        for artifact in artifacts:
            if "hidden_states_0" in artifact.included_tensor_names:
                return artifact.artifact_id
        return None


def _capture_phase_for_positions(
    positions: list[int],
    *,
    prompt_token_count: int,
    capture_mode: str,
    default_phase: str,
) -> str:
    if capture_mode == "replay":
        return "replay"
    if not positions:
        return default_phase
    prompt_positions = [position for position in positions if position < prompt_token_count]
    decode_positions = [position for position in positions if position >= prompt_token_count]
    if prompt_positions and decode_positions:
        return "mixed_prompt_prefill_decode"
    if prompt_positions:
        return "prompt_prefill"
    return "decode"


def _source_positions_for_capture(
    positions: list[int],
    *,
    prompt_token_count: int,
    capture_mode: str,
) -> list[int]:
    if capture_mode != "online":
        return positions
    return [position if position < prompt_token_count else position - 1 for position in positions]


def _hidden_state_position_semantics(
    *,
    prompt_token_count: int,
    generated_token_count: int,
    selected_positions: list[int],
    source_positions: list[int],
    capture_mode: str,
    capture_phase: str,
) -> dict[str, Any]:
    total_token_count = prompt_token_count + generated_token_count
    return {
        "position_index_space": "combined_prompt_then_generated_tokens",
        "prompt_span": [0, prompt_token_count],
        "generated_span": [prompt_token_count, total_token_count],
        "selected_positions": selected_positions,
        "source_positions": source_positions,
        "input_position_semantics": (
            "A hidden state at prompt/generated input position i is the representation after consuming token i."
        ),
        "decoder_only_prediction_semantics": (
            "For decoder-only generation, the representation that "
            "predicts a generated token may correspond to the "
            "previous input position."
        ),
        "online_generated_position_mapping": (
            "For capture_mode=online, generated-token selectors use the "
            "predictor/source position p - 1 because the original decode "
            "does not compute a hidden state after consuming the final "
            "generated token."
            if capture_mode == "online"
            else None
        ),
        "capture_mode": capture_mode,
        "capture_phase": capture_phase,
    }


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "torch"


def _select_positions(
    tensor: Any,
    positions: list[int],
    *,
    pool: str | None,
    param: str,
    token_context: dict[str, Any] | None = None,
) -> Any:
    if not positions:
        raise InvalidRequestError(
            "Hidden-state position selector resolved to no positions.",
            code="invalid_selector",
            param=param)
    if min(positions) < 0:
        raise UnsupportedExtractionError(
            "Captured hidden-state source positions cannot be negative.",
            code="hidden_state_position_unavailable",
            param=param,
            details={"positions": positions},
        )
    if _tensor_shape(tensor)[0] <= max(positions):
        shape = _tensor_shape(tensor)
        details = {
            "shape": shape,
            "captured_token_count": shape[0] if shape else 0,
            "positions": positions,
            "max_requested_position": max(positions),
        }
        if token_context is not None:
            details.update(token_context)
        raise UnsupportedExtractionError(
            "Captured hidden states do not cover all requested token positions.",
            code="hidden_state_position_unavailable",
            param=param,
            details=details,
        )
    if _is_torch_tensor(tensor):
        import torch

        index = torch.as_tensor(positions, dtype=torch.long, device=tensor.device)
        selected = tensor.index_select(0, index)
        if pool == "mean":
            return selected.mean(dim=0)
        if pool == "max":
            return selected.max(dim=0).values
        if pool == "last":
            return selected[-1]
        return selected

    array = np.asarray(tensor)
    selected = np.take(array, np.asarray(positions, dtype=np.int64), axis=0)
    if pool == "mean":
        return selected.mean(axis=0)
    if pool == "max":
        return selected.max(axis=0)
    if pool == "last":
        return selected[-1]
    return selected


def _stack_tensors(tensors: list[Any], *, param: str) -> Any:
    if not tensors:
        raise InvalidRequestError(
            "Hidden-state layer selector resolved to no layers.",
            code="invalid_selector",
            param=param)
    if all(_is_torch_tensor(tensor) for tensor in tensors):
        import torch

        return torch.stack(tensors, dim=0)
    return np.stack([_tensor_to_numpy(tensor) for tensor in tensors], axis=0)


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    if _is_torch_tensor(tensor):
        cpu_tensor = tensor.detach().cpu()
        try:
            return cpu_tensor.numpy()
        except TypeError:
            if getattr(cpu_tensor, "is_floating_point", lambda: False)():
                return cpu_tensor.float().numpy()
            raise
    return np.asarray(tensor)


def _tensor_to_jsonable(tensor: Any) -> Any:
    return _tensor_to_numpy(tensor).tolist()


def _tensor_shape(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    return [int(dim) for dim in np.asarray(tensor).shape]


def _tensor_dtype(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None:
        return str(dtype)
    return str(np.asarray(tensor).dtype)


def _tensor_device(tensor: Any) -> str:
    device = getattr(tensor, "device", None)
    return str(device) if device is not None else "cpu"


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if _is_torch_tensor(tensor):
        return int(tensor.nelement() * tensor.element_size())
    return int(np.asarray(tensor).nbytes)


def _tensor_storage_dtype(tensor: Any) -> str:
    return str(_tensor_to_numpy(tensor).dtype)


def _tensor_storage_nbytes(tensor: Any) -> int:
    return int(_tensor_to_numpy(tensor).nbytes)


def _tensor_artifact_nbytes(tensor: Any, format: str) -> int:
    if format == "npz":
        return _tensor_storage_nbytes(tensor)
    return _tensor_nbytes(tensor)
