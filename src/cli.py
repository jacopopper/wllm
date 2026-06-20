from __future__ import annotations

import argparse
import logging
from pathlib import Path


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wllm",
        description="vLLM serving with runtime-selectable white-box extraction.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the OpenAI-compatible server.")
    serve.add_argument("model", type=_nonempty_model, help="Model name or local model path to load with vLLM.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, bfloat16.")
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--max-model-len", type=int, default=None, help="Maximum model context length passed to vLLM.")
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
    serve.add_argument("--artifact-dir", default="./wllm-artifacts", help="Directory for trace artifacts.")
    serve.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    serve.set_defaults(func=_cmd_serve)
    return parser


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
        )
    )
    if runtime.config.prewarm_hidden_states:
        runtime.prewarm_hidden_states()
    app = create_app(
        runtime=runtime,
        artifact_store=ArtifactStore(Path(args.artifact_dir)),
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
