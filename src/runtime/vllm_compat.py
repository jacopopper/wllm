from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any

from packaging.version import Version

from extractors.planning import RuntimeTopology
from server.errors import RuntimeUnavailableError, UnsupportedExtractionError

SUPPORTED_VLLM_VERSION = Version("0.10.2")


@dataclass(frozen=True)
class VLLMImports:
    module: Any
    version: str
    LLM: Any
    SamplingParams: Any


@dataclass(frozen=True)
class OnlineHiddenStateCapture:
    output: Any
    tensors: dict[int, Any]
    capture_site: str
    capture_phase: str
    metadata: dict[str, Any]
    overhead_ms: float


def _apply_transformers_compat() -> None:
    """Apply compatibility shims for transformers >= 5.0 with vLLM 0.10.2.

    transformers 5.x removed PreTrainedTokenizerBase.all_special_tokens_extended,
    but vLLM 0.10.2 still accesses it. Restore it as a property that delegates
    to all_special_tokens (which returns the same list of token strings).
    """
    try:
        from transformers import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            # Attach a property that delegates to all_special_tokens.
            # In transformers 4.x all_special_tokens_extended returned
            # List[Union[str, AddedToken]]; vLLM treats the result as
            # a list of strings so the plain-string all_special_tokens
            # is compatible.
            def _get_all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
                return self.all_special_tokens

            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                _get_all_special_tokens_extended
            )
    except Exception:
        # If transformers isn't available, the vLLM import will fail
        # with its own error — let it.
        pass


def import_vllm() -> VLLMImports:
    try:
        version = importlib.metadata.version("vllm")
        parsed = Version(version)
        if parsed != SUPPORTED_VLLM_VERSION:
            raise RuntimeUnavailableError(
                f"Unsupported vLLM version {version}.",
                code="unsupported_vllm_version",
                details={
                    "supported": str(SUPPORTED_VLLM_VERSION),
                    "installed": version,
                },
            )
        # Compatibility shim: transformers >= 5.0 removed all_special_tokens_extended
        # which vLLM 0.10.2 still accesses. Restore it as an alias for all_special_tokens.
        _apply_transformers_compat()

        module = importlib.import_module("vllm")
        return VLLMImports(
            module=module,
            version=version,
            LLM=getattr(module, "LLM"),
            SamplingParams=getattr(module, "SamplingParams"),
        )
    except RuntimeUnavailableError:
        raise
    except Exception as exc:
        raise RuntimeUnavailableError(
            "vLLM is not installed or could not be imported. Install wllm with the vllm extra for production serving.",
            code="vllm_import_failed",
            details={"exception": repr(exc)},
        ) from exc


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameter_names = supported_parameter_names(callable_obj)
    if parameter_names is None:
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in parameter_names and value is not None}


def supported_parameter_names(callable_obj: Any) -> set[str] | None:
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return None
    return set(parameters)


def pooling_runner_kwargs(version: str | None) -> dict[str, Any]:
    common = {"enforce_eager": True, "enable_prefix_caching": False}
    if version is not None and Version(version) >= Version("0.10"):
        # vLLM 0.10 converts causal models to embedding models for generic
        # pooling. Its ALL pooler is encode-only, so use LAST for the public
        # pooler while hooks capture per-token transformer outputs upstream.
        return {
            "runner": "pooling",
            "convert": "embed",
            **common,
            "override_pooler_config": {"pooling_type": "LAST", "normalize": False},
        }
    return {"task": "embed", **common, "override_pooler_config": {"pooling_type": "ALL", "normalize": False}}


def pooling_runner_environment(version: str | None) -> dict[str, str]:
    if version is not None and Version(version) >= Version("0.10"):
        # vLLM 0.10's LLM.apply_model compatibility path requires an in-process
        # V1 engine. This is scoped to the isolated extraction runner only.
        return {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    return {}


def extract_model_topology(llm: Any) -> RuntimeTopology | None:
    """Best-effort topology read from vLLM public-ish model config objects.

    vLLM has changed where it stores the Hugging Face config across releases.
    This function is intentionally isolated so those probes do not spread through
    the runtime.
    """

    for config in _candidate_configs(llm):
        num_layers = _first_int_attr(
            config,
            [
                "num_hidden_layers",
                "n_layer",
                "num_layers",
                "n_layers",
            ],
        )
        if num_layers is None:
            continue
        return RuntimeTopology(
            num_layers=num_layers,
            num_attention_heads=_first_int_attr(
                config,
                [
                    "num_attention_heads",
                    "n_head",
                    "num_heads",
                    "n_heads",
                ],
            ),
            hidden_size=_first_int_attr(
                config,
                [
                    "hidden_size",
                    "n_embd",
                    "d_model",
                ],
            ),
        )
    return None


def extract_attention_backend(llm: Any, module: Any | None = None) -> str | None:
    """Best-effort attention backend read for capability reporting."""

    for env_name in ("VLLM_ATTENTION_BACKEND", "VLLM_USE_FLASHINFER"):
        value = os.environ.get(env_name)
        if value:
            return value
    for obj in [*_candidate_configs(llm), llm, module]:
        if obj is None:
            continue
        value = _first_str_attr(
            obj,
            [
                "attention_backend",
                "attn_backend",
                "selected_attention_backend",
                "_attention_backend",
            ],
        )
        if value:
            return value
    return None


def extract_supported_runner_types(llm: Any) -> list[str] | None:
    """Best-effort read of vLLM model_config.supported_runner_types."""

    for config in _candidate_configs(llm):
        value = getattr(config, "supported_runner_types", None)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        try:
            return [str(item) for item in value]
        except TypeError:
            continue
    return None


def capture_pooling_hidden_states(
    pooling_llm: Any,
    *,
    token_ids: list[int],
    layers: list[int],
) -> dict[int, Any]:
    """Capture selected transformer block outputs from an isolated pooling LLM.

    This intentionally uses temporary PyTorch module hooks only on the separate
    extraction runner, never on the normal serving runner. The caller is
    responsible for serializing access to the pooling runner.
    """

    if not hasattr(pooling_llm, "apply_model") or not hasattr(pooling_llm, "encode"):
        raise UnsupportedExtractionError(
            "The active vLLM version does not expose the model and encode surfaces required for scoped hidden-state capture.",
            code="hidden_states_unavailable",
            param="extract.hidden_states",
        )
    if not token_ids:
        raise UnsupportedExtractionError(
            "Hidden-state extraction requires at least one token in the final sequence.",
            code="hidden_state_positions_unavailable",
            param="extract.hidden_states",
        )
    unique_layers = _dedupe_ints(layers)
    try:
        install_results = _apply_model_to_workers(
            pooling_llm,
            lambda model: _install_hidden_state_hooks(model, unique_layers),
        )
    except Exception as exc:
        raise UnsupportedExtractionError(
            "Could not install scoped hidden-state capture hooks on the isolated vLLM pooling runner.",
            code="hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(exc)},
        ) from exc
    install_errors = [result for result in install_results if isinstance(result, dict) and result.get("error")]
    if install_errors:
        try:
            _apply_model_to_workers(pooling_llm, _pop_hidden_state_hooks)
        except Exception:
            pass
        raise UnsupportedExtractionError(
            "The isolated vLLM pooling runner does not expose compatible transformer block modules.",
            code="hidden_states_unavailable",
            param="extract.hidden_states",
            details={"errors": install_errors},
        )

    encode_error: Exception | None = None
    try:
        _encode_pooling_token_ids(pooling_llm, token_ids)
    except Exception as exc:
        encode_error = exc

    try:
        capture_results = _apply_model_to_workers(pooling_llm, _pop_hidden_state_hooks)
    except Exception as exc:
        if encode_error is not None:
            raise UnsupportedExtractionError(
                "The isolated vLLM pooling runner could not execute hidden-state capture for this model.",
                code="hidden_states_unavailable",
                param="extract.hidden_states",
                details={"exception": repr(encode_error), "cleanup_exception": repr(exc)},
            ) from encode_error
        raise UnsupportedExtractionError(
            "Could not remove scoped hidden-state capture hooks from the isolated vLLM pooling runner.",
            code="hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(exc)},
        ) from exc

    if encode_error is not None:
        raise UnsupportedExtractionError(
            "The isolated vLLM pooling runner could not execute hidden-state capture for this model.",
            code="hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(encode_error)},
        ) from encode_error

    captures: dict[int, Any] = {}
    missing = set(unique_layers)
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        for layer, tensor in result.items():
            layer_index = int(layer)
            captures[layer_index] = _combine_layer_captures(tensor)
            missing.discard(layer_index)
    if missing:
        raise UnsupportedExtractionError(
            "The isolated vLLM pooling runner did not capture every requested hidden-state layer.",
            code="hidden_state_layer_unavailable",
            param="extract.hidden_states",
            details={"missing_layers": sorted(missing), "captured_layers": sorted(captures)},
        )
    return captures


def capture_online_hidden_states(
    llm: Any,
    *,
    layers: list[int],
    generate: Any,
    capture_max_position: int | None = None,
) -> OnlineHiddenStateCapture:
    """Capture transformer block outputs from the active generation LLM.

    Hooks are installed immediately before the supplied generation callable and
    removed in a finally path. The returned metadata is best-effort because vLLM
    does not expose stable public prefill/decode hook boundaries here.
    """

    unique_layers = _dedupe_ints(layers)
    install_started = time.perf_counter()
    try:
        install_results = _apply_model_to_workers(
            llm,
            lambda model: _install_hidden_state_hooks(
                model,
                unique_layers,
                capture_max_position=capture_max_position,
            ),
        )
    except Exception as exc:
        try:
            _apply_model_to_workers(llm, _pop_hidden_state_hooks)
        except Exception:
            pass
        raise UnsupportedExtractionError(
            "Could not install scoped online hidden-state capture hooks on the active vLLM generation runner.",
            code="online_hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(exc)},
        ) from exc
    install_ms = (time.perf_counter() - install_started) * 1000.0

    install_errors = [result for result in install_results if isinstance(result, dict) and result.get("error")]
    if install_errors:
        try:
            _apply_model_to_workers(llm, _pop_hidden_state_hooks)
        except Exception:
            pass
        raise UnsupportedExtractionError(
            "The active vLLM generation runner does not expose compatible transformer block modules.",
            code="online_hidden_states_unavailable",
            param="extract.hidden_states",
            details={"errors": install_errors},
        )

    output: Any | None = None
    generate_error: Exception | None = None
    try:
        output = generate()
    except Exception as exc:
        generate_error = exc

    cleanup_started = time.perf_counter()
    try:
        capture_results = _apply_model_to_workers(llm, _pop_hidden_state_hooks)
    except Exception as exc:
        if generate_error is not None:
            raise UnsupportedExtractionError(
                "Online hidden-state capture cleanup failed after generation failed.",
                code="online_hidden_states_unavailable",
                param="extract.hidden_states",
                details={"exception": repr(generate_error), "cleanup_exception": repr(exc)},
            ) from generate_error
        raise UnsupportedExtractionError(
            "Could not remove scoped online hidden-state capture hooks from the active vLLM generation runner.",
            code="online_hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(exc)},
        ) from exc
    cleanup_ms = (time.perf_counter() - cleanup_started) * 1000.0

    if generate_error is not None:
        raise generate_error

    captures = _combine_capture_results(capture_results)
    missing = set(unique_layers) - set(captures)
    if missing:
        raise UnsupportedExtractionError(
            "The active vLLM generation runner did not capture every requested hidden-state layer.",
            code="online_hidden_state_layer_unavailable",
            param="extract.hidden_states",
            details={"missing_layers": sorted(missing), "captured_layers": sorted(captures)},
        )

    return OnlineHiddenStateCapture(
        output=output,
        tensors=captures,
        capture_site="transformer_block_output",
        capture_phase="prompt_prefill_decode_best_effort",
        metadata={
            "hook_scope": "active_generation_runner",
            "boundary_source": "forward_hook_chunk_shapes",
            "install_ms": install_ms,
            "cleanup_ms": cleanup_ms,
            "layer_chunk_shapes": _layer_chunk_shapes(capture_results),
            "capture_filter": _capture_filter_metadata(capture_max_position),
        },
        overhead_ms=install_ms + cleanup_ms,
    )


def _encode_pooling_token_ids(pooling_llm: Any, token_ids: list[int]) -> Any:
    kwargs: dict[str, Any] = {"use_tqdm": False}
    supported_tasks = getattr(pooling_llm, "supported_tasks", None)
    if supported_tasks is not None and "embed" in supported_tasks:
        kwargs["pooling_task"] = "embed"
    try:
        return pooling_llm.encode([{"prompt_token_ids": token_ids}], **kwargs)
    except TypeError as exc:
        try:
            return pooling_llm.encode(prompt_token_ids=[token_ids], **kwargs)
        except TypeError:
            raise exc


def _apply_model_to_workers(pooling_llm: Any, func: Any) -> list[Any]:
    engine = getattr(pooling_llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    if executor is not None and hasattr(executor, "apply_model"):
        return executor.apply_model(func)
    if hasattr(pooling_llm, "apply_model"):
        return pooling_llm.apply_model(func)
    raise AttributeError("The vLLM runner does not expose apply_model or model_executor.apply_model.")


def _candidate_configs(llm: Any) -> list[Any]:
    candidates: list[Any] = []
    direct_paths = [
        ("llm_engine", "model_config", "hf_config"),
        ("llm_engine", "model_config", "hf_text_config"),
        ("llm_engine", "model_config"),
        ("engine", "model_config", "hf_config"),
        ("engine", "model_config", "hf_text_config"),
        ("engine", "model_config"),
        ("model_config", "hf_config"),
        ("model_config", "hf_text_config"),
        ("model_config",),
    ]
    for path in direct_paths:
        value = _get_path(llm, path)
        if value is not None and value not in candidates:
            candidates.append(value)
    for value in list(candidates):
        for attr in ("hf_config", "hf_text_config", "text_config"):
            nested = getattr(value, attr, None)
            if nested is not None and nested not in candidates:
                candidates.append(nested)
    return candidates


def _get_path(root: Any, path: tuple[str, ...]) -> Any | None:
    value = root
    for attr in path:
        value = getattr(value, attr, None)
        if value is None:
            return None
    return value


def _first_int_attr(obj: Any, names: list[str]) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def _install_hidden_state_hooks(
    model: Any,
    layers: list[int],
    *,
    capture_max_position: int | None = None,
) -> dict[str, Any]:
    _remove_existing_hidden_state_hooks(model)
    modules = _locate_layer_modules(model)
    if modules is None:
        return {"error": "layer_modules_unavailable", "model_type": type(model).__name__}
    invalid = [layer for layer in layers if layer < 0 or layer >= len(modules)]
    if invalid:
        return {"error": "layer_index_out_of_range", "invalid_layers": invalid, "num_layers": len(modules)}
    state = {"captures": {}, "handles": [], "offsets": {}, "capture_max_position": capture_max_position}
    setattr(model, "_wllm_hidden_state_capture", state)
    for layer in layers:
        module = modules[layer]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer_index: int = layer) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                offset = int(state["offsets"].get(layer_index, 0))
                state["offsets"][layer_index] = offset + _tensor_row_count(tensor)
                tensor = _slice_capture_prefix(tensor, offset, state["capture_max_position"])
                if tensor is not None:
                    state["captures"].setdefault(layer_index, []).append(_to_cpu_tensor(tensor))

        state["handles"].append(module.register_forward_hook(hook))
    return {
        "installed_layers": layers,
        "num_layers": len(modules),
        "capture_filter": _capture_filter_metadata(capture_max_position),
    }


def _pop_hidden_state_hooks(model: Any) -> dict[int, Any]:
    state = getattr(model, "_wllm_hidden_state_capture", None)
    if not isinstance(state, dict):
        return {}
    captures = dict(state.get("captures", {}))
    _remove_existing_hidden_state_hooks(model)
    return captures


def _remove_existing_hidden_state_hooks(model: Any) -> None:
    state = getattr(model, "_wllm_hidden_state_capture", None)
    if isinstance(state, dict):
        for handle in state.get("handles", []):
            try:
                handle.remove()
            except Exception:
                pass
    if hasattr(model, "_wllm_hidden_state_capture"):
        try:
            delattr(model, "_wllm_hidden_state_capture")
        except Exception:
            setattr(model, "_wllm_hidden_state_capture", None)


def _locate_layer_modules(model: Any) -> Any | None:
    paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("bert", "encoder", "layer"),
    ]
    for path in paths:
        modules = _get_path(model, path)
        if modules is not None and hasattr(modules, "__len__") and hasattr(modules, "__getitem__"):
            return modules
    return None


def _first_tensor(value: Any) -> Any | None:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _to_cpu_tensor(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
        clone = getattr(tensor, "clone", None)
        if callable(clone):
            return clone()
        return tensor
    copy = getattr(tensor, "copy", None)
    if callable(copy):
        return copy()
    return tensor


def _tensor_row_count(tensor: Any) -> int:
    shape = getattr(tensor, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    import numpy as np

    return int(np.asarray(tensor).shape[0])


def _slice_capture_prefix(tensor: Any, offset: int, capture_max_position: int | None) -> Any | None:
    if capture_max_position is None:
        return tensor
    row_count = _tensor_row_count(tensor)
    if row_count <= 0 or offset > capture_max_position:
        return None
    keep = min(row_count, capture_max_position - offset + 1)
    if keep <= 0:
        return None
    if keep == row_count:
        return tensor
    return tensor[:keep]


def _combine_layer_captures(captures: Any) -> Any:
    if not isinstance(captures, list):
        return captures
    if not captures:
        return captures
    if len(captures) == 1:
        return captures[0]
    first = captures[0]
    if _is_torch_tensor(first) and all(_is_torch_tensor(item) for item in captures):
        import torch

        return torch.cat(captures, dim=0)
    import numpy as np

    return np.concatenate([np.asarray(item) for item in captures], axis=0)


def _combine_capture_results(capture_results: list[Any]) -> dict[int, Any]:
    per_layer: dict[int, list[Any]] = {}
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        for layer, captures in result.items():
            layer_index = int(layer)
            per_layer.setdefault(layer_index, []).append(_combine_layer_captures(captures))
    return {layer: _combine_layer_captures(captures) for layer, captures in per_layer.items()}


def _layer_chunk_shapes(capture_results: list[Any]) -> dict[str, list[list[int]]]:
    shapes: dict[str, list[list[int]]] = {}
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        for layer, captures in result.items():
            chunks = captures if isinstance(captures, list) else [captures]
            shapes.setdefault(str(int(layer)), []).extend(_shape_of_tensor(chunk) for chunk in chunks)
    return shapes


def _capture_filter_metadata(capture_max_position: int | None) -> dict[str, Any] | None:
    if capture_max_position is None:
        return None
    return {
        "type": "dense_prefix",
        "max_source_position": capture_max_position,
        "captured_source_positions": [0, capture_max_position],
    }


def _shape_of_tensor(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    import numpy as np

    return [int(dim) for dim in np.asarray(tensor).shape]


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "torch"


def _dedupe_ints(values: list[int]) -> list[int]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _first_str_attr(obj: Any, names: list[str]) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, (bool, int, float)):
            text = str(value)
            if text:
                return text
    return None
