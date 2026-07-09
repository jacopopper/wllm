"""vLLM compatibility layer for wllm.

This module is the single point of contact with private vLLM internals.
Every access to a non-public vLLM API, attribute, or data structure is
documented below with the observed vLLM version, expected shape, and
failure behavior when the underlying API changes.

**Supported vLLM version:** 0.10.2 (exact match enforced at import time).

=======================================================================
Catalog of private vLLM API dependencies
=======================================================================

Each entry lists the private API surface, the function using it, the
observed shape on vLLM 0.10.2, and expected failure mode if the API
changes in a future vLLM release.

-----------------------------------------------------------------------
1. ``LLM.apply_model(func)``
-----------------------------------------------------------------------
   **Accessed in:** ``_apply_model_to_workers()``
   **vLLM 0.10.2 shape:**
      Method on the model executor (``llm_engine.model_executor``)
      that runs a callable on every model worker and returns a list of
      per-worker results.  Also accessible directly on the ``LLM``
      instance as a convenience path.
   **Fallback chain:**
      a) ``pooling_llm.llm_engine.model_executor.apply_model(func)``
      b) ``pooling_llm.apply_model(func)``
   **Failure behavior:**
      If neither path exposes ``apply_model``, an ``AttributeError`` is
      raised at the call site and converted to
      ``UnsupportedExtractionError(code="hidden_states_unavailable")``
      or ``code="online_hidden_states_unavailable"``.

-----------------------------------------------------------------------
2. ``LLM.encode(...)``
-----------------------------------------------------------------------
   **Accessed in:** ``_encode_pooling_token_ids()``
   **vLLM 0.10.2 shape:**
      Method that accepts a list of dicts (``[{"prompt_token_ids": [...]}]``)
      and returns pooling output.  The ``pooling_task`` kwarg is set to
      ``"embed"`` when ``LLM.supported_tasks`` contains ``"embed"``.
   **Fallback:**
      Tries ``pooling_llm.encode([{"prompt_token_ids": token_ids}], ...)``
      first; if that raises ``TypeError``, attempts
      ``pooling_llm.encode(prompt_token_ids=[token_ids], ...)``.
   **Failure behavior:**
      If neither signature works, the original ``TypeError`` is
      re-raised, which the caller converts to
      ``UnsupportedExtractionError(code="hidden_states_unavailable")``.

-----------------------------------------------------------------------
3. ``LLM.supported_tasks``
-----------------------------------------------------------------------
   **Accessed in:** ``_encode_pooling_token_ids()``,
      ``extract_supported_runner_types()``
   **vLLM 0.10.2 shape:**
      An iterable of strings (e.g., ``["generate", "embed"]``) on the
      ``LLM`` instance.  Used to decide whether to pass
      ``pooling_task="embed"`` to ``encode()``.
   **Failure behavior:**
      Missing or ``None`` attribute is handled gracefully: the
      ``pooling_task`` kwarg is simply omitted.

-----------------------------------------------------------------------
4. ``LLM.llm_engine`` / ``LLM.engine``
-----------------------------------------------------------------------
   **Accessed in:** ``_apply_model_to_workers()``,
      ``_candidate_configs()``
   **vLLM 0.10.2 shape:**
      The internal vLLM engine object.  ``llm_engine`` is the primary
      attribute name; ``engine`` is probed as a fallback in config
      resolution paths.
   **Failure behavior:**
      ``getattr(pooling_llm, "llm_engine", None)`` returns ``None``
      when the attribute is absent; the callers handle ``None`` by
      skipping the dependent code path.

-----------------------------------------------------------------------
5. ``llm_engine.model_executor``
-----------------------------------------------------------------------
   **Accessed in:** ``_apply_model_to_workers()``
   **vLLM 0.10.2 shape:**
      The model executor object has an ``apply_model`` method (see #1).
   **Failure behavior:**
      ``AttributeError`` if ``model_executor`` is absent; handled at
      the ``_apply_model_to_workers`` level.

-----------------------------------------------------------------------
6. ``llm_engine.model_config`` / ``LLM.model_config``
-----------------------------------------------------------------------
   **Accessed in:** ``_candidate_configs()``,
      ``extract_model_topology()``, ``extract_attention_backend()``,
      ``extract_supported_runner_types()``
   **vLLM 0.10.2 shape:**
      A model configuration object containing HuggingFace config
      references.  Probed through multiple paths:
        - ``llm_engine.model_config``
        - ``engine.model_config``
        - ``model_config`` (directly on LLM)
   **Failure behavior:**
      Missing attribute → silently skipped in ``_candidate_configs()``;
      topology extraction returns ``None`` when no config path yields
      layer/head/hidden_size counts.

-----------------------------------------------------------------------
7. ``model_config.hf_config`` / ``model_config.hf_text_config``
-----------------------------------------------------------------------
   **Accessed in:** ``_candidate_configs()``,
      ``extract_model_topology()``
   **vLLM 0.10.2 shape:**
      The underlying HuggingFace model config objects.  ``hf_config``
      is the primary accessor; ``hf_text_config`` is probed for VLMs
      with separate text configs.  Nested ``hf_config`` /
      ``text_config`` attributes are also probed.
   **Failure behavior:**
      Missing → the config object itself is used as fallback for
      topology attribute probing; if it lacks the expected layer/head
      attributes, topology extraction returns ``None``.

-----------------------------------------------------------------------
8. Model layer module access paths
-----------------------------------------------------------------------
   **Accessed in:** ``_locate_layer_modules()``
   **vLLM 0.10.2 shape:**
      Sequential module container probed through these attribute paths
      (in order):
        - ``model.layers`` (most common: Llama, Mistral, Qwen2)
        - ``model.model.layers`` (nested wrapper)
        - ``language_model.model.layers`` (multi-modal)
        - ``transformer.h`` (GPT-2 / older)
        - ``gpt_neox.layers`` (GPT-NeoX)
        - ``bert.encoder.layer`` (BERT-family)
      Must be indexable (``__len__`` + ``__getitem__``).
   **Failure behavior:**
      If no path resolves, returns ``None``; the caller reports
      ``"error": "layer_modules_unavailable"``.

-----------------------------------------------------------------------
9. HuggingFace config topology attributes
-----------------------------------------------------------------------
   **Accessed in:** ``extract_model_topology()``
   **vLLM 0.10.2 shape (on the HuggingFace config object):**
      - Layer count: ``num_hidden_layers`` → ``n_layer`` → ``num_layers`` → ``n_layers``
      - Attention heads: ``num_attention_heads`` → ``n_head`` → ``num_heads`` → ``n_heads``
      - Hidden size: ``hidden_size`` → ``n_embd`` → ``d_model``
      Each probe returns the first attribute whose value is a positive
      ``int`` (``bool`` values are rejected).
   **Failure behavior:**
      Missing all candidates → ``RuntimeTopology`` field is ``None``.
      Topology extraction returns ``None`` when any required field
      (num_layers) is ``None``.

-----------------------------------------------------------------------
10. ``model_config.supported_runner_types``
-----------------------------------------------------------------------
   **Accessed in:** ``extract_supported_runner_types()``
   **vLLM 0.10.2 shape:**
      A string or iterable of strings (e.g., ``"generate"``,
      ``["generate", "pooling"]``) on the model config.
   **Failure behavior:**
      Missing → returns ``None``.

-----------------------------------------------------------------------
11. Attention backend detection attributes
-----------------------------------------------------------------------
   **Accessed in:** ``extract_attention_backend()``
   **vLLM 0.10.2 shape:**
      Probed through environment variables first
      (``VLLM_ATTENTION_BACKEND``, ``VLLM_USE_FLASHINFER``), then
      through these object attributes on LLM and config objects:
        - ``attention_backend``
        - ``attn_backend``
        - ``selected_attention_backend``
        - ``_attention_backend``
   **Failure behavior:**
      All probes miss → returns ``None``.

-----------------------------------------------------------------------
12. ``LLM.get_tokenizer()``
-----------------------------------------------------------------------
   **Accessed in:** ``vllm_runtime.py`` (uses public API)
   **vLLM 0.10.2 shape:**
      Returns a HuggingFace tokenizer instance.  This is a public API
      on vLLM's ``LLM`` class.  The compat module does not access it
      directly, but the runtime relies on it for chat template
      rendering and token decoding.

-----------------------------------------------------------------------
13. ``PreTrainedTokenizerBase.all_special_tokens_extended`` (transformers compat)
-----------------------------------------------------------------------
   **Accessed in:** ``_apply_transformers_compat()``
   **Context:**
      transformers >= 5.0 removed ``all_special_tokens_extended``,
      but vLLM 0.10.2 still accesses it.  We restore it as a property
      delegating to ``all_special_tokens``.
   **Failure behavior:**
      If ``transformers`` is not importable, the compat shim is skipped
      silently; the vLLM import will fail with its own error.

=======================================================================
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import time
from dataclasses import dataclass, field
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
    selected_tensors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineHiddenStateSelection:
    name: str
    layers: list[int]
    positions: list[int]
    pool: str | None = None


@dataclass(frozen=True)
class TransformersAttentionReplay:
    model: Any
    tokenizer: Any | None
    torch: Any
    device: str


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

            PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
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

    **Private API accessed:**
    HuggingFace config attributes on vLLM's model config (vLLM 0.10.2):
    ``num_hidden_layers`` / ``n_layer`` / ``num_layers`` / ``n_layers``,
    ``num_attention_heads`` / ``n_head`` / ``num_heads`` / ``n_heads``,
    ``hidden_size`` / ``n_embd`` / ``d_model``.
    Config resolution paths documented in ``_candidate_configs()``.

    **Expected shape:**
    Returns a ``RuntimeTopology`` with integer ``num_layers``,
    ``num_attention_heads``, and ``hidden_size`` fields.  Any field may
    be ``None`` if the corresponding config attribute is absent.

    **Failure behavior:**
    Returns ``None`` when ``num_layers`` cannot be resolved (the
    minimum required field).  Missing ``num_attention_heads`` or
    ``hidden_size`` produce ``None`` for those fields on the returned
    topology object but do not prevent topology extraction.

    vLLM has changed where it stores the Hugging Face config across
    releases.  This function is intentionally isolated so those probes
    do not spread through the runtime.
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
    """Best-effort attention backend read for capability reporting.

    **Private API accessed:**
    Environment variables ``VLLM_ATTENTION_BACKEND`` and
    ``VLLM_USE_FLASHINFER`` (checked first).  Then probes config and
    LLM objects (vLLM 0.10.2) for these attributes:
    ``attention_backend``, ``attn_backend``,
    ``selected_attention_backend``, ``_attention_backend``.

    **Expected shape:** a non-empty string (e.g., ``"FLASH_ATTN"``,
    ``"FLASHINFER"``) or ``None``.

    **Failure behavior:** returns ``None`` when no backend is
    discoverable through any probe.  This is non-fatal; capability
    reporting degrades gracefully.
    """

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
    """Best-effort read of vLLM model_config.supported_runner_types.

    **Private API accessed:**
    ``model_config.supported_runner_types`` on vLLM 0.10.2 config objects
    (probed through ``_candidate_configs()`` paths).

    **Expected shape:** a list of strings (e.g., ``["generate", "pooling"]``)
    or a single string.  Returns ``None`` when the attribute is absent.

    **Failure behavior:** returns ``None`` gracefully.  Callers treat
    ``None`` as "unknown" and do not gate critical paths on this value.
    """

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


def load_transformers_attention_replay(
    *,
    model: str,
    tokenizer: str | None,
    dtype: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> TransformersAttentionReplay:
    """Load the separate Transformers model used for replay attention capture."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise UnsupportedExtractionError(
            "Attention replay requires torch and transformers to be installed.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={"exception": repr(exc), "requires": ["torch", "transformers"]},
        ) from exc

    common_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    model_kwargs = dict(common_kwargs)
    torch_dtype = _transformers_torch_dtype(torch, dtype)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    model_kwargs["attn_implementation"] = "eager"
    try:
        model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    except Exception as exc:
        raise UnsupportedExtractionError(
            "Could not initialize the Transformers model for attention replay.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={
                "exception": repr(exc),
                "model": model,
                "dtype": dtype,
                "local_files_only": local_files_only,
            },
        ) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_obj.to(device)
    except Exception as exc:
        raise UnsupportedExtractionError(
            "Could not move the Transformers attention replay model to the selected device.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={"exception": repr(exc), "model": model, "device": str(device)},
        ) from exc
    model_obj.eval()
    return TransformersAttentionReplay(model=model_obj, tokenizer=None, torch=torch, device=str(device))


def capture_transformers_replay_attentions(
    replay: TransformersAttentionReplay,
    *,
    token_ids: list[int],
) -> dict[int, Any]:
    """Replay a final token sequence through Transformers with output_attentions=True."""

    if not token_ids:
        raise UnsupportedExtractionError(
            "Attention extraction requires at least one token in the final sequence.",
            code="attention_positions_unavailable",
            param="extract.attentions",
        )
    torch = replay.torch
    device = _model_device(replay.model, torch)
    try:
        input_ids = torch.as_tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = replay.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:
        raise UnsupportedExtractionError(
            "The Transformers replay model could not produce attention weights for this sequence.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={"exception": repr(exc), "token_count": len(token_ids)},
        ) from exc

    attentions = getattr(outputs, "attentions", None)
    if attentions is None and isinstance(outputs, dict):
        attentions = outputs.get("attentions")
    if not attentions:
        raise UnsupportedExtractionError(
            "The Transformers replay model did not return attention tensors.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={"token_count": len(token_ids)},
        )

    captures: dict[int, Any] = {}
    for layer_index, tensor in enumerate(attentions):
        if tensor is None:
            continue
        shape = [int(dim) for dim in getattr(tensor, "shape", [])]
        if len(shape) == 4 and shape[0] == 1:
            tensor = tensor[0]
            shape = [int(dim) for dim in getattr(tensor, "shape", [])]
        if len(shape) != 3:
            raise UnsupportedExtractionError(
                "The Transformers replay model returned attention tensors with an unsupported shape.",
                code="attention_weights_unavailable",
                param="extract.attentions",
                details={"layer": layer_index, "shape": shape},
            )
        captures[layer_index] = tensor.detach()
    if not captures:
        raise UnsupportedExtractionError(
            "The Transformers replay model returned no usable attention tensors.",
            code="attention_weights_unavailable",
            param="extract.attentions",
            details={"token_count": len(token_ids)},
        )
    return captures


# ---------------------------------------------------------------------------
# Raw logits capture via logits processor (for exact entropy, margins, etc.)
# ---------------------------------------------------------------------------

def create_raw_logits_processor() -> tuple[callable, list]:
    """Return a logits processor + list that will receive full raw logits tensors.

    The processor is suitable for SamplingParams.logits_processors.
    Each call (one per generated token) appends a CPU clone of the full
    (vocab_size,) logits tensor for that position.

    This gives access to the un-normalized logits that vLLM used for sampling,
    enabling exact entropy and other full-distribution UQ methods.
    """
    captured: list = []

    def _raw_logits_processor(
        past_token_ids: list[int], logits: "torch.Tensor"
    ) -> "torch.Tensor":
        # logits is (vocab_size,) or (batch, vocab) depending on context
        # For n=1 it is usually 1D per step in the way processors are called.
        if hasattr(logits, "dim") and logits.dim() > 1:
            logits = logits[0]
        captured.append(logits.detach().cpu().clone())
        return logits

    return _raw_logits_processor, captured


def capture_pooling_hidden_states(
    pooling_llm: Any,
    *,
    token_ids: list[int],
    layers: list[int],
    site: str = "block",
) -> dict[int, Any]:
    """Capture selected transformer block outputs from an isolated pooling LLM.

    This intentionally uses temporary PyTorch module hooks only on the separate
    extraction runner, never on the normal serving runner. The caller is
    responsible for serializing access to the pooling runner.
    """

    if not hasattr(pooling_llm, "apply_model") or not hasattr(pooling_llm, "encode"):
        raise UnsupportedExtractionError(
            "The active vLLM version does not expose the model and encode surfaces "
            "required for scoped hidden-state capture.",
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
            lambda model: _install_hidden_state_hooks(model, unique_layers, site=site),
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
    select_hidden_states: Any | None = None,
    site: str = "block",
) -> OnlineHiddenStateCapture:
    """Capture transformer block outputs from the active generation LLM.

    Hooks are installed immediately before the supplied generation callable and
    removed in a finally path. The returned metadata is best-effort because vLLM
    does not expose stable public prefill/decode hook boundaries here.

    When ``select_hidden_states`` is supplied, it is called with the generation
    output after generation completes and must return
    ``OnlineHiddenStateSelection`` objects. Selection and pooling then happen on
    the model worker before tensors are copied back to the request process.
    """

    unique_layers = _dedupe_ints(layers)
    copy_to_cpu = select_hidden_states is None
    install_started = time.perf_counter()
    try:
        install_results = _apply_model_to_workers(
            llm,
            lambda model: _install_hidden_state_hooks(
                model,
                unique_layers,
                capture_max_position=capture_max_position,
                copy_to_cpu=copy_to_cpu,
                site=site,
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

    selections: list[OnlineHiddenStateSelection] = []
    selection_error: Exception | None = None
    if generate_error is None and select_hidden_states is not None:
        try:
            selections = list(select_hidden_states(output))
        except Exception as exc:
            selection_error = exc

    cleanup_started = time.perf_counter()
    try:
        if selection_error is None and select_hidden_states is not None:
            capture_results = _apply_model_to_workers(
                llm,
                lambda model: _pop_selected_hidden_state_hooks(model, selections),
            )
        else:
            capture_results = _apply_model_to_workers(llm, _pop_hidden_state_hooks)
    except Exception as exc:
        if generate_error is not None or selection_error is not None:
            primary_error = generate_error or selection_error
            raise UnsupportedExtractionError(
                "Online hidden-state capture cleanup failed after generation or selector planning failed.",
                code="online_hidden_states_unavailable",
                param="extract.hidden_states",
                details={"exception": repr(primary_error), "cleanup_exception": repr(exc)},
            ) from primary_error
        raise UnsupportedExtractionError(
            "Could not remove scoped online hidden-state capture hooks from the active vLLM generation runner.",
            code="online_hidden_states_unavailable",
            param="extract.hidden_states",
            details={"exception": repr(exc)},
        ) from exc
    cleanup_ms = (time.perf_counter() - cleanup_started) * 1000.0

    if generate_error is not None:
        raise generate_error
    if selection_error is not None:
        raise selection_error

    selected_tensors: dict[str, Any] = {}
    if select_hidden_states is None:
        captures = _combine_capture_results(capture_results)
        layer_chunk_shapes = _layer_chunk_shapes(capture_results)
        missing = set(unique_layers) - set(captures)
        if missing:
            raise UnsupportedExtractionError(
                "The active vLLM generation runner did not capture every requested hidden-state layer.",
                code="online_hidden_state_layer_unavailable",
                param="extract.hidden_states",
                details={"missing_layers": sorted(missing), "captured_layers": sorted(captures)},
            )
    else:
        captures = {}
        layer_chunk_shapes = _selected_capture_layer_chunk_shapes(capture_results)
        selected_errors = [result for result in capture_results if isinstance(result, dict) and result.get("error")]
        if selected_errors:
            raise UnsupportedExtractionError(
                "The active vLLM generation runner could not select requested hidden-state tensors.",
                code="online_hidden_states_unavailable",
                param="extract.hidden_states",
                details={"errors": selected_errors},
            )
        selected_tensors = _combine_selected_capture_results(capture_results)
        missing = {selection.name for selection in selections} - set(selected_tensors)
        if missing:
            raise UnsupportedExtractionError(
                "The active vLLM generation runner did not return every selected hidden-state tensor.",
                code="online_hidden_state_selection_unavailable",
                param="extract.hidden_states",
                details={"missing_tensors": sorted(missing), "captured_tensors": sorted(selected_tensors)},
            )

    return OnlineHiddenStateCapture(
        output=output,
        tensors=captures,
        capture_site="block",
        capture_phase="prompt_prefill_decode_best_effort",
        metadata={
            "hook_scope": "active_generation_runner",
            "boundary_source": "forward_hook_chunk_shapes",
            "install_ms": install_ms,
            "cleanup_ms": cleanup_ms,
            "layer_chunk_shapes": layer_chunk_shapes,
            "selected_tensor_shapes": _selected_tensor_shapes(selected_tensors),
            "selection_mode": "worker_selected" if select_hidden_states is not None else "raw_capture",
            "capture_filter": _capture_filter_metadata(capture_max_position),
        },
        overhead_ms=install_ms + cleanup_ms,
        selected_tensors=selected_tensors,
    )


def _encode_pooling_token_ids(pooling_llm: Any, token_ids: list[int]) -> Any:
    """Execute a pooling encode of *token_ids* on the isolated vLLM runner.

    **Private API accessed:**
    ``LLM.encode(...)`` (vLLM 0.10.2).
    ``LLM.supported_tasks`` (vLLM 0.10.2).

    **Expected shape:**
    Primary call signature: ``pooling_llm.encode([{"prompt_token_ids": token_ids}], use_tqdm=False)``.
    If ``LLM.supported_tasks`` contains ``"embed"``, also passes
    ``pooling_task="embed"``.
    Fallback signature (on ``TypeError``): ``pooling_llm.encode(prompt_token_ids=[token_ids], ...)``.

    **Failure behavior:**
    ``TypeError`` from both signatures is re-raised; the caller
    (``capture_pooling_hidden_states``) converts it to
    ``UnsupportedExtractionError(code="hidden_states_unavailable")``.
    """
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
    """Apply *func* to every model worker through the private vLLM executor.

    **Private API accessed:**
    ``LLM.llm_engine.model_executor.apply_model(func)`` (vLLM 0.10.2).
    Falls back to ``LLM.apply_model(func)`` if the executor path is absent.

    **Expected shape:** returns a ``list`` of per-worker results (one entry
    per tensor-parallel worker).  Hooks installed by *func* must return
    dicts keyed by layer index.

    **Failure behavior:** ``AttributeError`` if neither path exposes
    ``apply_model``.  Callers convert this to an
    ``UnsupportedExtractionError`` with a clear message.
    """
    engine = getattr(pooling_llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    if executor is not None and hasattr(executor, "apply_model"):
        return executor.apply_model(func)
    if hasattr(pooling_llm, "apply_model"):
        return pooling_llm.apply_model(func)
    raise AttributeError("The vLLM runner does not expose apply_model or model_executor.apply_model.")


def _candidate_configs(llm: Any) -> list[Any]:
    """Collect all reachable model-config objects from a vLLM ``LLM`` instance.

    **Private API accessed:**
    vLLM internal config layout (vLLM 0.10.2).  Probes these attribute
    paths on the ``LLM`` instance:

    ===============================  =====================================
    Path                             Notes
    ===============================  =====================================
    ``llm_engine.model_config.hf_config``        Primary HF config
    ``llm_engine.model_config.hf_text_config``   VLM text-only config
    ``llm_engine.model_config``                  Fallback
    ``engine.model_config.hf_config``            Alternate engine attr
    ``engine.model_config.hf_text_config``       Alternate engine attr
    ``engine.model_config``                      Alternate engine fallback
    ``model_config.hf_config``                   Direct model_config
    ``model_config.hf_text_config``              Direct model_config
    ``model_config``                             Direct fallback
    ===============================  =====================================

    Additionally, each discovered config is inspected for nested
    ``hf_config``, ``hf_text_config``, and ``text_config`` attributes.

    **Expected shape:** a list of config objects (may contain duplicates;
    deduplication is done by identity).  The list may be empty.

    **Failure behavior:** missing attributes are silently skipped via
    ``getattr(..., None)``.  An empty list means no config was reachable.
    """
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
    copy_to_cpu: bool = True,
    site: str = "block",
) -> dict[str, Any]:
    """Install forward hooks on selected transformer-block layers.

    **Private API accessed:**
    PyTorch ``register_forward_hook`` (public API) on layer modules
    discovered via ``_locate_layer_modules()`` (vLLM-private module
    layout, vLLM 0.10.2).

    **Expected shape:**
    Each hook intercepts the forward pass output, extracts the first
    tensor, detaches it, and appends it to a per-layer
    capture list stored on ``model._wllm_hidden_state_capture``.
    ``copy_to_cpu=True`` preserves the legacy raw-capture behavior.
    ``copy_to_cpu=False`` keeps tensors on the worker until selector
    compaction runs in ``_pop_selected_hidden_state_hooks``.

    Returns a dict with ``installed_layers``, ``num_layers``, and
    optional ``capture_filter`` metadata.

    **Failure behavior:**
    Returns ``{"error": "layer_modules_unavailable", ...}`` when
    ``_locate_layer_modules`` returns ``None``.
    Returns ``{"error": "layer_index_out_of_range", ...}`` when any
    requested layer index is outside ``[0, num_layers)``.
    The caller converts these to ``UnsupportedExtractionError``.
    """
    _remove_existing_hidden_state_hooks(model)
    modules = _locate_layer_modules(model)
    if modules is None:
        return {"error": "layer_modules_unavailable", "model_type": type(model).__name__}
    invalid = [layer for layer in layers if layer < 0 or layer >= len(modules)]
    if invalid:
        return {"error": "layer_index_out_of_range", "invalid_layers": invalid, "num_layers": len(modules)}
    state: dict[str, Any] = {
        "captures": {},
        "handles": [],
        "offsets": {},
        "chunk_shapes": {},
        "capture_max_position": capture_max_position,
        "copy_to_cpu": copy_to_cpu,
    }
    setattr(model, "_wllm_hidden_state_capture", state)
    for layer_idx in layers:
        layer_mod = modules[layer_idx]
        target = _get_layer_target_module(layer_mod, site)
        module = target if target is not None else layer_mod

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer_index: int = layer_idx) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                offset = int(state["offsets"].get(layer_index, 0))
                state["offsets"][layer_index] = offset + _tensor_row_count(tensor)
                tensor = _slice_capture_prefix(tensor, offset, state["capture_max_position"])
                if tensor is not None:
                    state["chunk_shapes"].setdefault(layer_index, []).append(_shape_of_tensor(tensor))
                    captured = _to_cpu_tensor(tensor) if state["copy_to_cpu"] else _detach_tensor(tensor)
                    state["captures"].setdefault(layer_index, []).append(captured)

        state["handles"].append(module.register_forward_hook(hook))
    return {
        "installed_layers": layers,
        "num_layers": len(modules),
        "capture_filter": _capture_filter_metadata(capture_max_position),
        "site": site,
    }


def _pop_hidden_state_hooks(model: Any) -> dict[int, Any]:
    state = getattr(model, "_wllm_hidden_state_capture", None)
    if not isinstance(state, dict):
        return {}
    captures = dict(state.get("captures", {}))
    _remove_existing_hidden_state_hooks(model)
    return captures


def _pop_selected_hidden_state_hooks(
    model: Any,
    selections: list[OnlineHiddenStateSelection],
) -> dict[str, Any]:
    state = getattr(model, "_wllm_hidden_state_capture", None)
    if not isinstance(state, dict):
        return {
            "error": "hidden_state_capture_state_unavailable",
            "requested_tensors": [selection.name for selection in selections],
        }
    captures = dict(state.get("captures", {}))
    chunk_shapes = {
        str(int(layer)): [list(shape) for shape in shapes]
        for layer, shapes in dict(state.get("chunk_shapes", {})).items()
    }
    try:
        selected = {
            selection.name: _select_hidden_state_capture(captures, selection)
            for selection in selections
        }
        return {
            "__selected_tensors__": selected,
            "__layer_chunk_shapes__": chunk_shapes,
        }
    except Exception as exc:
        return {
            "error": "hidden_state_selection_failed",
            "exception": repr(exc),
            "requested_tensors": [selection.name for selection in selections],
            "layer_chunk_shapes": chunk_shapes,
        }
    finally:
        _remove_existing_hidden_state_hooks(model)


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
    """Find the sequential transformer-block container on a vLLM model.

    **Private API accessed:**
    vLLM model internal layout.  Probing paths (vLLM 0.10.2):

    ===========================  ==============================
    Architecture pattern         Attribute path
    ===========================  ==============================
    Llama / Mistral / Qwen2      ``model.layers``
    Nested wrapper               ``model.model.layers``
    Multi-modal                  ``language_model.model.layers``
    GPT-2 / older                ``transformer.h``
    GPT-NeoX                     ``gpt_neox.layers``
    BERT-family                  ``bert.encoder.layer``
    ===========================  ==============================

    **Expected shape:** a ``ModuleList`` or equivalent sequential container
    supporting ``__len__`` and ``__getitem__``.

    **Failure behavior:** returns ``None`` when no path resolves.  The
    caller reports ``"error": "layer_modules_unavailable"`` with the
    model type name, which surfaces as
    ``UnsupportedExtractionError(code="hidden_states_unavailable")``.
    """
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


def _get_layer_target_module(layer: Any, site: str = "block") -> Any | None:
    """Return the appropriate submodule inside a transformer layer for the capture site.

    Supports:
      - "block": the full layer (default, post residual)
      - "post_attn": the attention submodule output
      - "post_mlp": the MLP/FFN submodule output

    This enables richer hidden-state capture for probing and analysis methods.
    """
    if site == "block" or not site:
        return layer

    # Common attribute names across architectures (Llama, Mistral, Qwen, etc.)
    if site == "post_attn":
        for name in ("self_attn", "attn", "attention", "self_attention"):
            mod = getattr(layer, name, None)
            if mod is not None:
                return mod
    elif site == "post_mlp":
        for name in ("mlp", "feed_forward", "ffn", "mlp_block"):
            mod = getattr(layer, name, None)
            if mod is not None:
                return mod

    # Fallback to the full layer
    return layer


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


def _detach_tensor(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
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


def _select_hidden_state_capture(
    captures: dict[int, Any],
    selection: OnlineHiddenStateSelection,
) -> Any:
    selected_layers = []
    for layer in selection.layers:
        layer_captures = captures.get(layer)
        if layer_captures is None:
            raise ValueError(f"missing captured hidden states for layer {layer}")
        layer_tensor = _combine_layer_captures(layer_captures)
        selected_layers.append(
            _select_positions_from_tensor(
                layer_tensor,
                positions=selection.positions,
                pool=selection.pool,
            )
        )
    return _to_cpu_tensor(_stack_tensors(selected_layers))


def _select_positions_from_tensor(tensor: Any, *, positions: list[int], pool: str | None) -> Any:
    if not positions:
        raise ValueError("hidden-state position selector resolved to no positions")
    if min(positions) < 0:
        raise ValueError(f"hidden-state source positions cannot be negative: {positions!r}")
    shape = _shape_of_tensor(tensor)
    if not shape or shape[0] <= max(positions):
        raise ValueError(
            "captured hidden states do not cover requested source positions: "
            f"shape={shape!r} positions={positions!r}"
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

    import numpy as np

    array = np.asarray(tensor)
    selected = np.take(array, np.asarray(positions, dtype=np.int64), axis=0)
    if pool == "mean":
        return selected.mean(axis=0)
    if pool == "max":
        return selected.max(axis=0)
    if pool == "last":
        return selected[-1]
    return selected


def _stack_tensors(tensors: list[Any]) -> Any:
    if not tensors:
        raise ValueError("hidden-state layer selector resolved to no layers")
    if all(_is_torch_tensor(tensor) for tensor in tensors):
        import torch

        return torch.stack(tensors, dim=0)
    import numpy as np

    return np.stack([np.asarray(tensor) for tensor in tensors], axis=0)


def _combine_selected_capture_results(capture_results: list[Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        tensors = result.get("__selected_tensors__")
        if isinstance(tensors, dict):
            selected.update(tensors)
    return selected


def _selected_capture_layer_chunk_shapes(capture_results: list[Any]) -> dict[str, list[list[int]]]:
    shapes: dict[str, list[list[int]]] = {}
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        worker_shapes = result.get("__layer_chunk_shapes__")
        if not isinstance(worker_shapes, dict):
            continue
        for layer, chunks in worker_shapes.items():
            shapes.setdefault(str(layer), []).extend([list(chunk) for chunk in chunks])
    return shapes


def _layer_chunk_shapes(capture_results: list[Any]) -> dict[str, list[list[int]]]:
    shapes: dict[str, list[list[int]]] = {}
    for result in capture_results:
        if not isinstance(result, dict):
            continue
        for layer, captures in result.items():
            chunks = captures if isinstance(captures, list) else [captures]
            shapes.setdefault(str(int(layer)), []).extend(_shape_of_tensor(chunk) for chunk in chunks)
    return shapes


def _selected_tensor_shapes(selected_tensors: dict[str, Any]) -> dict[str, list[int]]:
    return {name: _shape_of_tensor(tensor) for name, tensor in selected_tensors.items()}


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


def _transformers_torch_dtype(torch: Any, dtype: str) -> Any | None:
    normalized = dtype.lower()
    if normalized in {"float16", "half"}:
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized in {"float32", "float"}:
        return torch.float32
    return None


def _model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


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
