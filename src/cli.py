from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any


def _nonempty_model(value: str) -> str:
    """argparse type-check rejecting empty model identifiers.

    Accepts HuggingFace IDs (``org/name``), absolute filesystem paths, and
    relative filesystem paths. An empty string is rejected so that
    ``wllm serve ""`` fails with a clear argparse error rather than silently
    proceeding to vLLM initialization.
    """
    if not value:
        raise argparse.ArgumentTypeError("model must be a non-empty HuggingFace ID or path")
    return value


def _positive_int(value: str) -> int:
    """argparse type that rejects non-positive integer values (<= 0)."""
    try:
        ival = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"invalid positive int value: {value!r}")
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive, got {ival}")
    return ival


def _valid_port(value: str) -> int:
    """argparse type that rejects out-of-range port numbers."""
    try:
        ival = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"invalid port value: {value!r}")
    if ival < 1 or ival > 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {ival}")
    return ival


def _fraction(value: str) -> float:
    """argparse type that rejects values outside the usable (0.0, 1.0] range."""
    try:
        fval = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}")
    if fval <= 0.0 or fval > 1.0:
        raise argparse.ArgumentTypeError(f"value must be greater than 0.0 and at most 1.0, got {fval}")
    return fval


def _resource_limits_from_args(args: argparse.Namespace):
    from extractors.planning import ResourceLimits

    return ResourceLimits(
        max_top_k=args.max_top_k,
        max_selected_layers=args.max_selected_layers,
        max_selected_heads=args.max_selected_heads,
        max_selected_positions=args.max_selected_positions,
        max_inline_tensor_bytes=args.max_inline_tensor_bytes,
        max_total_captured_tensor_bytes=args.max_total_captured_tensor_bytes,
        max_artifact_bytes=args.max_artifact_bytes,
        large_extraction_enabled=args.large_extraction_enabled,
    )


def _path_like_model(value: str) -> bool:
    return value.startswith((".", "/", "~")) or "\\" in value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wllm",
        description="vLLM serving with runtime-selectable white-box extraction.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the OpenAI-compatible server.")
    serve.add_argument("model", type=_nonempty_model, help="Model name or local model path to load with vLLM.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve.add_argument("--port", type=_valid_port, default=8000, help="Bind port.")
    serve.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, bfloat16.")
    serve.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    serve.add_argument("--gpu-memory-utilization", type=_fraction, default=0.9)
    serve.add_argument(
        "--max-model-len", type=_positive_int, default=None,
        help="Maximum model context length passed to vLLM.",
    )
    serve.add_argument("--tokenizer", default=None, help="Tokenizer name or path passed to vLLM.")
    serve.add_argument("--served-model-name", default=None, help="Model name returned by /v1/models and in responses.")
    serve.add_argument("--api-key", default=None, help="Optional API key required on all requests.")
    serve.add_argument("--seed", type=int, default=None, help="Default seed for sampling parameters.")
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--local-files-only", action="store_true", help="Prevent network model downloads.")
    serve.add_argument(
        "--prewarm-hidden-states",
        action="store_true",
        help="Initialize the optional hidden-state extraction runner before accepting requests.",
    )
    serve.add_argument(
        "--enable-online-hidden-states",
        action="store_true",
        help="Enable capture_mode=online using an eager in-process vLLM generation runner.",
    )
    serve.add_argument(
        "--enable-attention-weights",
        action="store_true",
        help="Enable experimental replay-only attention weight extraction on /v1/extract and /v1/traces.",
    )
    serve.add_argument("--artifact-dir", default="./wllm-artifacts", help="Directory for trace artifacts.")
    serve.add_argument("--max-top-k", type=_positive_int, default=50, help="Maximum extract.logprobs.top_k.")
    serve.add_argument(
        "--max-selected-layers",
        type=_positive_int,
        default=8,
        help="Maximum selected layers per tensor request unless large extraction is enabled.",
    )
    serve.add_argument(
        "--max-selected-heads",
        type=_positive_int,
        default=32,
        help="Maximum selected attention heads per request unless large extraction is enabled.",
    )
    serve.add_argument(
        "--max-selected-positions",
        type=_positive_int,
        default=256,
        help="Maximum selected token positions per tensor request unless large extraction is enabled.",
    )
    serve.add_argument(
        "--max-inline-tensor-bytes",
        type=_positive_int,
        default=1_000_000,
        help="Maximum estimated tensor bytes returned inline in a trace response.",
    )
    serve.add_argument(
        "--max-total-captured-tensor-bytes",
        type=_positive_int,
        default=64_000_000,
        help="Maximum estimated tensor bytes captured for one extraction request.",
    )
    serve.add_argument(
        "--max-artifact-bytes",
        type=_positive_int,
        default=256_000_000,
        help="Maximum serialized byte size for one artifact or trace bundle.",
    )
    serve.add_argument(
        "--enable-large-extraction",
        dest="large_extraction_enabled",
        action="store_true",
        help="Allow artifact-backed requests that exceed default selector-count limits.",
    )
    serve.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    serve.set_defaults(func=_cmd_serve)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check the local wllm/vLLM environment without starting a model.",
        description="Check the local wllm/vLLM environment without starting a model.",
    )
    doctor.add_argument("--model", default=None, help="Optional local model path or Hugging Face ID to sanity-check.")
    doctor.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require a path-like --model to exist locally; Hugging Face cache state is reported as unverified.",
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = _doctor_checks(model=args.model, local_files_only=args.local_files_only)
    has_error = any(check["status"] == "error" for check in checks)
    if args.json:
        print(json.dumps({"ok": not has_error, "checks": checks}, sort_keys=True))
    else:
        print("wllm doctor")
        for check in checks:
            status = str(check["status"])
            name = str(check["name"])
            message = str(check["message"])
            print(f"[{status}] {name}: {message}")
    return 1 if has_error else 0


def _doctor_checks(*, model: str | None, local_files_only: bool) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "python",
            "status": "ok",
            "message": f"{platform.python_version()} on {platform.system() or sys.platform}",
            "version": platform.python_version(),
        },
        _package_check("wllm"),
        _vllm_version_check(),
        _optional_package_check("torch", feature="PT artifacts and replay attention"),
        _optional_package_check("transformers", feature="replay attention"),
    ]
    if model is not None:
        checks.append(_model_check(model, local_files_only=local_files_only))
    return checks


def _package_check(distribution: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": distribution,
            "status": "warning",
            "message": "package metadata is not installed; editable source-tree execution may still work",
            "version": None,
        }
    return {"name": distribution, "status": "ok", "message": f"installed version {version}", "version": version}


def _vllm_version_check() -> dict[str, Any]:
    supported = "0.10.2"
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": "vllm",
            "status": "error",
            "message": "vLLM is not installed; install wllm with the vllm extra for serving",
            "version": None,
            "supported": supported,
        }
    if version != supported:
        return {
            "name": "vllm",
            "status": "error",
            "message": f"unsupported version {version}; expected exactly {supported}",
            "version": version,
            "supported": supported,
        }
    return {
        "name": "vllm",
        "status": "ok",
        "message": f"installed supported version {version}",
        "version": version,
        "supported": supported,
    }


def _optional_package_check(distribution: str, *, feature: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": distribution,
            "status": "warning",
            "message": f"not installed; required only for {feature}",
            "version": None,
        }
    return {
        "name": distribution,
        "status": "ok",
        "message": f"installed version {version}",
        "version": version,
    }


def _model_check(model: str, *, local_files_only: bool) -> dict[str, Any]:
    expanded = Path(model).expanduser()
    if _path_like_model(model):
        if expanded.exists():
            return {
                "name": "model",
                "status": "ok",
                "message": f"local path exists: {expanded}",
                "model": model,
            }
        return {
            "name": "model",
            "status": "error" if local_files_only else "warning",
            "message": f"local path does not exist: {expanded}",
            "model": model,
        }
    return {
        "name": "model",
        "status": "warning" if local_files_only else "ok",
        "message": (
            "Hugging Face cache state is not verified by doctor; --local-files-only will prevent downloads"
            if local_files_only
            else "Hugging Face model ID format; cache/download state is not verified by doctor"
        ),
        "model": model,
    }


def _cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Keep vLLM imports lazy so `wllm --help` and module imports remain cheap.
    from artifacts.store import ArtifactStore
    from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
    from server.app import create_app
    import uvicorn

    runtime = VLLMRuntime(
        VLLMRuntimeConfig(
            model=args.model,
            served_model_name=args.served_model_name,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tokenizer=args.tokenizer,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            prewarm_hidden_states=args.prewarm_hidden_states,
            enable_online_hidden_states=args.enable_online_hidden_states,
            enable_attention_weights=args.enable_attention_weights,
        )
    )
    if runtime.config.prewarm_hidden_states:
        runtime.prewarm_hidden_states()
    app = create_app(
        runtime=runtime,
        artifact_store=ArtifactStore(Path(args.artifact_dir)),
        limits=_resource_limits_from_args(args),
        api_key=args.api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if argv is not None:
            return int(exc.code or 0)
        raise
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
