from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class Case:
    name: str
    engine: str
    description: str
    extraction: str | None = None
    hidden_layers: str | int | list[int] | None = None
    hidden_positions: str | int | list[int] | None = None
    hidden_capture_mode: str | None = None
    artifact_format: str | None = None
    artifact_compression: str | None = None
    persist_trace: bool = False
    enable_online_hidden_states: bool = False
    enable_attention_weights: bool = False
    attention_layers: str | int | list[int] | None = None
    attention_heads: str | int | list[int] | None = None
    attention_query_positions: str | int | list[int] | None = None
    attention_key_positions: str | int | list[int] | None = None


SCENARIOS = {
    "short": Scenario(
        name="short",
        prompt="State one practical reason to inspect hidden states during safety evaluation.",
        max_tokens=8,
    ),
    "medium": Scenario(
        name="medium",
        prompt=(
            "A safety researcher wants deterministic serving, token logprobs, and selected activation traces. "
            "Explain the latency tradeoffs and what should stay off the inference hot path."
        ),
        max_tokens=32,
    ),
    "long": Scenario(
        name="long",
        prompt=(
            "Benchmark request: compare clean generation, lightweight tracing, logprob extraction, selected hidden "
            "state capture, and artifact persistence for a white-box language model serving stack. Focus on "
            "operational latency, GPU transfer volume, and which work can be deferred or handled asynchronously. "
        )
        * 4,
        max_tokens=64,
    ),
}


CASES = {
    "raw_vllm.generate": Case(
        name="raw_vllm.generate",
        engine="raw_vllm",
        description="Direct vLLM LLM.generate with use_tqdm disabled.",
    ),
    "transformers.generate": Case(
        name="transformers.generate",
        engine="transformers",
        description="Hugging Face Transformers generate baseline, if installed and runnable.",
    ),
    "wllm.completion": Case(
        name="wllm.completion",
        engine="wllm_completion",
        description="wllm OpenAI-compatible completion path without extraction.",
    ),
    "wllm.extract.tokens": Case(
        name="wllm.extract.tokens",
        engine="wllm_extract",
        extraction="tokens",
        description="wllm extraction with token IDs and decoded tokens only.",
    ),
    "wllm.extract.logprobs": Case(
        name="wllm.extract.logprobs",
        engine="wllm_extract",
        extraction="logprobs",
        description="wllm extraction with tokens, top-k logprobs, prompt logprobs, and approximate entropy.",
    ),
    "wllm.persist.tokens_logprobs_npz": Case(
        name="wllm.persist.tokens_logprobs_npz",
        engine="wllm_extract",
        extraction="tokens_logprobs_npz",
        artifact_format="npz",
        persist_trace=True,
        description="wllm persisted trace plus token/logprob NPZ artifact.",
    ),
    "wllm.hidden.replay.last_inline": Case(
        name="wllm.hidden.replay.last_inline",
        engine="wllm_extract",
        extraction="hidden",
        hidden_layers="middle",
        hidden_positions="last_generated",
        hidden_capture_mode="replay",
        description="wllm replay hidden-state capture, one middle layer, last generated position inline.",
    ),
    "wllm.hidden.replay.generated_npz": Case(
        name="wllm.hidden.replay.generated_npz",
        engine="wllm_extract",
        extraction="hidden",
        hidden_layers="middle",
        hidden_positions="generated",
        hidden_capture_mode="replay",
        artifact_format="npz",
        artifact_compression="uncompressed",
        description="wllm replay hidden-state capture for generated positions saved as uncompressed NPZ.",
    ),
    "wllm.hidden.online.last_inline": Case(
        name="wllm.hidden.online.last_inline",
        engine="wllm_extract",
        extraction="hidden",
        hidden_layers="middle",
        hidden_positions="last_generated",
        hidden_capture_mode="online",
        enable_online_hidden_states=True,
        description="wllm online hidden-state capture, one middle layer, last generated position inline.",
    ),
    "wllm.hidden.online.generated_npz": Case(
        name="wllm.hidden.online.generated_npz",
        engine="wllm_extract",
        extraction="hidden",
        hidden_layers="middle",
        hidden_positions="generated",
        hidden_capture_mode="online",
        artifact_format="npz",
        artifact_compression="uncompressed",
        enable_online_hidden_states=True,
        description="wllm online hidden-state capture for generated positions saved as uncompressed NPZ.",
    ),
    "wllm.hidden.online.middle_third_generated_npz": Case(
        name="wllm.hidden.online.middle_third_generated_npz",
        engine="wllm_extract",
        extraction="hidden",
        hidden_layers="middle_third",
        hidden_positions="generated",
        hidden_capture_mode="online",
        artifact_format="npz",
        artifact_compression="uncompressed",
        enable_online_hidden_states=True,
        description="wllm online hidden-state capture for middle-third layers and generated positions.",
    ),
    "wllm.attention.replay.last_inline": Case(
        name="wllm.attention.replay.last_inline",
        engine="wllm_extract",
        extraction="attentions",
        enable_attention_weights=True,
        attention_layers="middle",
        attention_heads=[0, 1],
        attention_query_positions="last_generated",
        attention_key_positions="previous_token",
        description="wllm replay attention capture, one middle layer and two heads for the last generated token.",
    ),
    "wllm.attention.replay.generated_npz": Case(
        name="wllm.attention.replay.generated_npz",
        engine="wllm_extract",
        extraction="attentions",
        enable_attention_weights=True,
        attention_layers="middle",
        attention_heads=[0, 1],
        attention_query_positions="generated",
        attention_key_positions="previous_token",
        artifact_format="npz",
        artifact_compression="uncompressed",
        description="wllm replay attention capture for generated positions saved as uncompressed NPZ.",
    ),
}


PROFILE_CASES = {
    "quick": [
        "raw_vllm.generate",
        "wllm.completion",
        "wllm.extract.tokens",
        "wllm.extract.logprobs",
        "wllm.hidden.online.last_inline",
        "wllm.attention.replay.last_inline",
    ],
    "standard": [
        "raw_vllm.generate",
        "wllm.completion",
        "wllm.extract.tokens",
        "wllm.extract.logprobs",
        "wllm.persist.tokens_logprobs_npz",
        "wllm.hidden.replay.last_inline",
        "wllm.hidden.online.last_inline",
        "wllm.hidden.online.generated_npz",
        "wllm.attention.replay.last_inline",
        "wllm.attention.replay.generated_npz",
    ],
    "full": [
        "raw_vllm.generate",
        "transformers.generate",
        "wllm.completion",
        "wllm.extract.tokens",
        "wllm.extract.logprobs",
        "wllm.persist.tokens_logprobs_npz",
        "wllm.hidden.replay.last_inline",
        "wllm.hidden.replay.generated_npz",
        "wllm.hidden.online.last_inline",
        "wllm.hidden.online.generated_npz",
        "wllm.hidden.online.middle_third_generated_npz",
        "wllm.attention.replay.last_inline",
        "wllm.attention.replay.generated_npz",
    ],
}


PROFILE_SCENARIOS = {
    "quick": ["short"],
    "standard": ["short", "medium"],
    "full": ["short", "medium", "long"],
}


PROFILE_DEFAULTS = {
    "quick": {"warmups": 1, "runs": 3},
    "standard": {"warmups": 1, "runs": 6},
    "full": {"warmups": 2, "runs": 10},
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run_child(args), sort_keys=True, default=str))
        return 0
    return run_parent(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Latency suite for raw vLLM, optional Transformers, and wllm extraction paths."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_CASES), default="quick")
    parser.add_argument("--cases", help="Comma-separated cases. Defaults to the selected profile.")
    parser.add_argument("--scenarios", help="Comma-separated scenarios: short, medium, long.")
    parser.add_argument("--prompt", help="Override scenarios with one custom prompt.")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens for every scenario.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tokenizer")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-total-captured-tensor-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--max-inline-tensor-bytes", type=int, default=64_000_000)
    parser.add_argument("--max-artifact-bytes", type=int, default=4_000_000_000)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "wllm-latency-artifacts",
    )
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "wllm_latency_results.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "wllm_latency_report.md")
    parser.add_argument(
        "--gpu-memory-wait-mib",
        type=int,
        default=2048,
        help="Before and after each child case, wait until nvidia-smi reports GPU memory below this MiB threshold.",
    )
    parser.add_argument("--gpu-memory-wait-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--case-retries",
        type=int,
        default=1,
        help="Retry a failed child case this many times before recording it as failed.",
    )
    parser.add_argument(
        "--retry-sleep-s",
        type=float,
        default=5.0,
        help="Sleep this many seconds between failed child-case attempts.",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=sorted(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help=argparse.SUPPRESS)
    return parser


def run_parent(args: argparse.Namespace) -> int:
    args.warmups = profile_default(args, "warmups")
    args.runs = profile_default(args, "runs")
    case_names = parse_names(args.cases, PROFILE_CASES[args.profile], allowed=CASES, label="case")
    scenario_names = parse_names(args.scenarios, PROFILE_SCENARIOS[args.profile], allowed=SCENARIOS, label="scenario")
    if args.prompt is not None:
        scenario_names = ["custom"]

    jobs = [(case_name, scenario_name) for scenario_name in scenario_names for case_name in case_names]
    if args.dry_run:
        print(json.dumps({"jobs": [{"case": case, "scenario": scenario} for case, scenario in jobs]}, indent=2))
        return 0

    results = []
    for case_name, scenario_name in jobs:
        result = run_child_with_retries(args, case_name, scenario_name)
        results.append(result)
        status = "ok" if "error" not in result else "failed"
        print(f"{scenario_name} / {case_name}: {status}", file=sys.stderr, flush=True)
        wait_for_gpu_memory_below(args.gpu_memory_wait_mib, timeout_s=args.gpu_memory_wait_timeout_s)

    payload = {
        "schema_version": "wllm.latency_suite.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": parent_config(args, case_names, scenario_names),
        "environment": environment_summary(),
        "results": add_baseline_ratios(results),
    }
    write_json(args.output_json, payload)
    write_text(args.report, render_markdown_report(payload))
    print(json.dumps({"output_json": str(args.output_json), "report": str(args.report)}, sort_keys=True))
    has_errors = any("error" in result for result in results)
    return 1 if has_errors and args.fail_on_error else 0


def run_child_with_retries(args: argparse.Namespace, case_name: str, scenario_name: str) -> dict[str, Any]:
    max_attempts = max(int(args.case_retries), 0) + 1
    failed_attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        wait_for_gpu_memory_below(args.gpu_memory_wait_mib, timeout_s=args.gpu_memory_wait_timeout_s)
        completed = subprocess.run(
            child_command(args, case_name, scenario_name),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        result = parse_child_result(case_name, scenario_name, completed)
        result["attempts"] = attempt
        if "error" not in result:
            if failed_attempts:
                result["failed_attempts"] = failed_attempts
            return result

        failed_attempts.append(compact_failed_attempt(attempt, result))
        if attempt < max_attempts:
            print(
                f"{scenario_name} / {case_name}: failed attempt {attempt}/{max_attempts}, retrying",
                file=sys.stderr,
                flush=True,
            )
            wait_for_gpu_memory_below(args.gpu_memory_wait_mib, timeout_s=args.gpu_memory_wait_timeout_s)
            if args.retry_sleep_s > 0:
                time.sleep(float(args.retry_sleep_s))

    assert result is not None
    result["failed_attempts"] = failed_attempts
    return result


def parse_child_result(
    case_name: str,
    scenario_name: str,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    result = parse_child_json(completed.stdout)
    if result is None:
        return {
            "case": case_name,
            "scenario": scenario_name,
            "error": "child process did not emit JSON",
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    if completed.returncode != 0:
        result["error"] = result.get("error") or f"child process exited with {completed.returncode}"
        result["stderr_tail"] = completed.stderr[-4000:]
    return result


def compact_failed_attempt(attempt: int, result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "attempt": attempt,
        "error": result.get("error"),
        "error_code": result.get("error_code"),
        "returncode": result.get("returncode"),
    }
    if result.get("stderr_tail"):
        compact["stderr_tail"] = str(result["stderr_tail"])[-1000:]
    if result.get("traceback"):
        compact["traceback_tail"] = str(result["traceback"])[-1000:]
    return {key: value for key, value in compact.items() if value is not None}


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.case is None:
            raise ValueError("--case is required in child mode")
        case = CASES[args.case]
        scenario = scenario_from_args(args)
        set_offline_environment(args)
        runner, load_ms = make_runner(args, case, scenario)
        warmups = [runner() for _ in range(profile_default(args, "warmups"))]
        samples = [runner() for _ in range(profile_default(args, "runs"))]
        return {
            "case": case.name,
            "scenario": scenario.name,
            "description": case.description,
            "config": child_config(args, scenario),
            "environment": environment_summary(),
            "load_ms": load_ms,
            "warmups": warmups,
            "samples": samples,
            "summary": summarize_samples(samples),
        }
    except Exception as exc:
        return {
            "case": args.case,
            "scenario": args.scenario,
            "config": vars(args),
            "environment": environment_summary(),
            "error": f"{type(exc).__name__}: {exc}",
            "error_details": getattr(exc, "details", None),
            "error_code": getattr(exc, "code", None),
            "error_param": getattr(exc, "param", None),
            "traceback": traceback.format_exc(),
        }


def child_command(args: argparse.Namespace, case_name: str, scenario_name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--case",
        case_name,
        "--scenario",
        scenario_name if scenario_name != "custom" else "short",
        "--model",
        args.model,
        "--profile",
        args.profile,
        "--batch-size",
        str(args.batch_size),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-total-captured-tensor-bytes",
        str(args.max_total_captured_tensor_bytes),
        "--max-inline-tensor-bytes",
        str(args.max_inline_tensor_bytes),
        "--max-artifact-bytes",
        str(args.max_artifact_bytes),
        "--artifact-root",
        str(args.artifact_root),
        "--case-retries",
        str(args.case_retries),
        "--retry-sleep-s",
        str(args.retry_sleep_s),
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.tokenizer is not None:
        command.extend(["--tokenizer", args.tokenizer])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.local_files_only:
        command.append("--local-files-only")
    if args.prompt is not None:
        command.extend(["--prompt", args.prompt])
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])
    return command


def make_runner(args: argparse.Namespace, case: Case, scenario: Scenario) -> tuple[Any, float]:
    if case.engine == "raw_vllm":
        return make_raw_vllm_runner(args, scenario)
    if case.engine == "transformers":
        return make_transformers_runner(args, scenario)
    if case.engine == "wllm_completion":
        return make_wllm_completion_runner(args, scenario)
    if case.engine == "wllm_extract":
        return make_wllm_extract_runner(args, case, scenario)
    raise ValueError(f"Unknown case engine {case.engine!r}")


def make_raw_vllm_runner(args: argparse.Namespace, scenario: Scenario) -> tuple[Any, float]:
    from runtime.vllm_compat import _apply_transformers_compat

    _apply_transformers_compat()
    from vllm import LLM, SamplingParams

    prompts = [scenario.prompt for _ in range(args.batch_size)]
    kwargs = vllm_kwargs(args)
    started = time.perf_counter()
    llm = LLM(**kwargs)
    load_ms = elapsed_ms(started)
    sampling = SamplingParams(max_tokens=scenario.max_tokens, temperature=0.0, seed=args.seed)

    def run_once() -> dict[str, Any]:
        started = time.perf_counter()
        outputs = generate_with_tqdm_disabled(llm, prompts, sampling)
        wall_ms = elapsed_ms(started)
        prompt_tokens, generated_tokens = usage_from_vllm_outputs(outputs)
        return run_record(wall_ms, prompt_tokens, generated_tokens, request_count=len(prompts))

    return run_once, load_ms


def make_transformers_runner(args: argparse.Namespace, scenario: Scenario) -> tuple[Any, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.dtype in {"auto", "float16", "half"} and torch.cuda.is_available() else None
    model_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    load_ms = elapsed_ms(started)
    prompts = [scenario.prompt for _ in range(args.batch_size)]

    def run_once() -> dict[str, Any]:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_tokens = int(encoded["attention_mask"].sum().item())
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=scenario.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        wall_ms = elapsed_ms(started)
        generated_tokens = max(int(output_ids.numel()) - int(encoded["input_ids"].numel()), 0)
        return run_record(wall_ms, prompt_tokens, generated_tokens, request_count=len(prompts))

    return run_once, load_ms


def make_wllm_completion_runner(args: argparse.Namespace, scenario: Scenario) -> tuple[Any, float]:
    from runtime.vllm_runtime import VLLMRuntime
    from schemas.openai import CompletionRequest

    started = time.perf_counter()
    runtime = VLLMRuntime(wllm_config(args, enable_online_hidden_states=False))
    runtime._ensure_loaded()
    load_ms = elapsed_ms(started)
    prompts = [scenario.prompt for _ in range(args.batch_size)]

    def run_once() -> dict[str, Any]:
        request = CompletionRequest(
            model=args.model,
            prompt=prompts,
            max_tokens=scenario.max_tokens,
            temperature=0.0,
            seed=args.seed,
        )
        started = time.perf_counter()
        response = runtime.generate_completion(request)
        wall_ms = elapsed_ms(started)
        usage = response["usage"]
        return run_record(
            wall_ms,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            request_count=len(prompts),
        )

    return run_once, load_ms


def make_wllm_extract_runner(args: argparse.Namespace, case: Case, scenario: Scenario) -> tuple[Any, float]:
    from artifacts.store import ArtifactStore
    from extractors.planning import ResourceLimits
    from runtime.vllm_runtime import VLLMRuntime

    started = time.perf_counter()
    runtime = VLLMRuntime(
        wllm_config(
            args,
            enable_online_hidden_states=case.enable_online_hidden_states,
            enable_attention_weights=case.enable_attention_weights,
        )
    )
    runtime._ensure_loaded()
    load_ms = elapsed_ms(started)
    artifact_store = ArtifactStore(args.artifact_root / f"{case.name}-{scenario.name}-{os.getpid()}")
    limits = ResourceLimits(
        max_inline_tensor_bytes=args.max_inline_tensor_bytes,
        max_total_captured_tensor_bytes=args.max_total_captured_tensor_bytes,
        max_artifact_bytes=args.max_artifact_bytes,
        large_extraction_enabled=True,
    )
    request = extraction_request(args, case, scenario)

    def run_once() -> dict[str, Any]:
        started = time.perf_counter()
        prompt_tokens = 0
        generated_tokens = 0
        artifact_bytes = 0
        artifact_count = 0
        trace_timing: dict[str, float] = {}
        for _ in range(args.batch_size):
            trace = runtime.generate_extract(
                request,
                limits=limits,
                artifact_store=artifact_store,
                persist=case.persist_trace,
            )
            usage = trace.generation["usage"]
            prompt_tokens += int(usage["prompt_tokens"])
            generated_tokens += int(usage["completion_tokens"])
            artifact_bytes += sum(int(artifact.byte_size) for artifact in trace.artifacts)
            artifact_count += len(trace.artifacts)
            if trace.trace_manifest is not None:
                artifact_bytes += int(trace.trace_manifest.byte_size)
                artifact_count += 1
            trace_timing = add_numeric_dicts(trace_timing, trace.metadata.timing_ms.model_dump(mode="json"))
        wall_ms = elapsed_ms(started)
        record = run_record(wall_ms, prompt_tokens, generated_tokens, request_count=args.batch_size)
        record["artifact_bytes"] = artifact_bytes
        record["artifact_count"] = artifact_count
        record["trace_timing_ms"] = trace_timing
        return record

    return run_once, load_ms


def extraction_request(args: argparse.Namespace, case: Case, scenario: Scenario) -> Any:
    from schemas.extraction import ExtractRequest

    extract: dict[str, Any]
    if case.extraction == "tokens":
        extract = {"tokens": True}
    elif case.extraction == "logprobs":
        extract = {
            "tokens": True,
            "logprobs": {
                "top_k": args.top_k,
                "include_prompt": True,
                "entropy": True,
                "allow_approximate_entropy": True,
            },
        }
    elif case.extraction == "tokens_logprobs_npz":
        extract = {
            "tokens": True,
            "logprobs": {"top_k": args.top_k, "include_prompt": True},
            "artifacts": {"format": "npz", "include": ["tokens", "logprobs"]},
        }
    elif case.extraction == "hidden":
        hidden = {
            "layers": case.hidden_layers,
            "positions": case.hidden_positions,
            "capture_mode": case.hidden_capture_mode,
        }
        extract = {"hidden_states": [hidden]}
        if case.artifact_format is not None:
            artifact: dict[str, Any] = {
                "format": case.artifact_format,
                "include": ["hidden_states"],
                "allow_large": True,
            }
            if case.artifact_compression is not None:
                artifact["compression"] = case.artifact_compression
            extract["artifacts"] = artifact
    elif case.extraction == "attentions":
        attention = {
            "layers": case.attention_layers,
            "heads": case.attention_heads,
            "query_positions": case.attention_query_positions,
            "key_positions": case.attention_key_positions,
        }
        extract = {"attentions": [attention]}
        if case.artifact_format is not None:
            artifact = {
                "format": case.artifact_format,
                "include": ["attentions"],
                "allow_large": True,
            }
            if case.artifact_compression is not None:
                artifact["compression"] = case.artifact_compression
            extract["artifacts"] = artifact
    else:
        raise ValueError(f"Case {case.name!r} does not define an extraction request")
    return ExtractRequest.model_validate(
        {
            "model": args.model,
            "prompt": scenario.prompt,
            "max_tokens": scenario.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "extract": extract,
        }
    )


def wllm_config(
    args: argparse.Namespace,
    *,
    enable_online_hidden_states: bool,
    enable_attention_weights: bool = False,
) -> Any:
    from runtime.vllm_runtime import VLLMRuntimeConfig

    return VLLMRuntimeConfig(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tokenizer=args.tokenizer,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        enable_online_hidden_states=enable_online_hidden_states,
        enable_attention_weights=enable_attention_weights,
    )


def vllm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    return {key: value for key, value in values.items() if value is not None}


def generate_with_tqdm_disabled(llm: Any, prompts: list[str], sampling: Any) -> Any:
    try:
        return llm.generate(prompts, sampling, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, sampling)


def usage_from_vllm_outputs(outputs: list[Any]) -> tuple[int, int]:
    prompt_tokens = 0
    generated_tokens = 0
    for output in outputs:
        prompt_tokens += len(getattr(output, "prompt_token_ids", []) or [])
        for completion in getattr(output, "outputs", []) or []:
            generated_tokens += len(getattr(completion, "token_ids", []) or [])
    return prompt_tokens, generated_tokens


def run_record(
    wall_ms: float,
    prompt_tokens: int,
    generated_tokens: int,
    *,
    request_count: int,
) -> dict[str, Any]:
    return {
        "wall_ms": wall_ms,
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": prompt_tokens + generated_tokens,
        "generated_tokens_per_s": generated_tokens / (wall_ms / 1000.0) if wall_ms > 0 else None,
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "wall_ms": stats([float(sample["wall_ms"]) for sample in samples]),
        "generated_tokens_per_s": stats(
            [
                float(sample["generated_tokens_per_s"])
                for sample in samples
                if sample.get("generated_tokens_per_s") is not None
            ]
        ),
        "artifact_bytes": stats([float(sample.get("artifact_bytes", 0.0)) for sample in samples]),
    }
    trace_keys = sorted({key for sample in samples for key in sample.get("trace_timing_ms", {})})
    if trace_keys:
        summary["trace_timing_ms"] = {
            key: stats([float(sample.get("trace_timing_ms", {}).get(key, 0.0)) for sample in samples])
            for key in trace_keys
        }
    return summary


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def percentile(ordered_values: list[float], fraction: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = (len(ordered_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = index - lower
    return ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight


def add_numeric_dicts(total: dict[str, float], values: dict[str, Any]) -> dict[str, float]:
    for key, value in values.items():
        if isinstance(value, int | float):
            total[key] = total.get(key, 0.0) + float(value)
    return total


def add_baseline_ratios(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        result["scenario"]: nested(result, "summary", "wall_ms", "median")
        for result in results
        if result.get("case") == "raw_vllm.generate" and "error" not in result
    }
    updated = []
    for result in results:
        result = dict(result)
        baseline = baselines.get(result.get("scenario"))
        median = nested(result, "summary", "wall_ms", "median")
        if baseline and median:
            result["median_wall_vs_raw_vllm"] = float(median) / float(baseline)
        updated.append(result)
    return updated


def scenario_from_args(args: argparse.Namespace) -> Scenario:
    if args.prompt is not None:
        max_tokens = args.max_tokens if args.max_tokens is not None else SCENARIOS[args.scenario or "short"].max_tokens
        return Scenario(name="custom", prompt=args.prompt, max_tokens=max_tokens)
    if args.scenario is None:
        raise ValueError("--scenario is required in child mode")
    base = SCENARIOS[args.scenario]
    if args.max_tokens is None:
        return base
    return Scenario(name=base.name, prompt=base.prompt, max_tokens=args.max_tokens)


def profile_default(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    return int(PROFILE_DEFAULTS[args.profile][name])


def parse_names(
    raw: str | None,
    default: list[str],
    *,
    allowed: dict[str, Any],
    label: str,
) -> list[str]:
    names = default if raw is None else [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(names) - set(allowed))
    if invalid:
        raise SystemExit(f"Unknown {label} names: {', '.join(invalid)}")
    return names


def parse_child_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def set_offline_environment(args: argparse.Namespace) -> None:
    if not args.local_files_only:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def parent_config(args: argparse.Namespace, case_names: list[str], scenario_names: list[str]) -> dict[str, Any]:
    config = common_config(args)
    config.update(
        {
            "cases": case_names,
            "scenarios": scenario_names,
            "profile": args.profile,
            "warmups": args.warmups,
            "runs": args.runs,
            "output_json": str(args.output_json),
            "report": str(args.report),
        }
    )
    return config


def child_config(args: argparse.Namespace, scenario: Scenario) -> dict[str, Any]:
    config = common_config(args)
    config.update(
        {
            "scenario": scenario.name,
            "prompt_chars": len(scenario.prompt),
            "max_tokens": scenario.max_tokens,
            "warmups": profile_default(args, "warmups"),
            "runs": profile_default(args, "runs"),
        }
    )
    return config


def common_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "local_files_only": args.local_files_only,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_total_captured_tensor_bytes": args.max_total_captured_tensor_bytes,
        "max_inline_tensor_bytes": args.max_inline_tensor_bytes,
        "max_artifact_bytes": args.max_artifact_bytes,
        "artifact_root": str(args.artifact_root),
        "gpu_memory_wait_mib": args.gpu_memory_wait_mib,
        "gpu_memory_wait_timeout_s": args.gpu_memory_wait_timeout_s,
        "case_retries": args.case_retries,
        "retry_sleep_s": args.retry_sleep_s,
    }


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "vllm_version": module_version("vllm"),
        "torch_version": module_version("torch"),
        "transformers_version": module_version("transformers"),
        "cuda": cuda_summary(),
        "gpu": nvidia_smi_summary(),
    }


def module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def cuda_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def nvidia_smi_summary() -> str | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return completed.stderr.strip() or None
    return completed.stdout.strip()


def wait_for_gpu_memory_below(threshold_mib: int | None, *, timeout_s: float) -> None:
    if threshold_mib is None or threshold_mib <= 0:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = current_gpu_memory_used_mib()
        if used is None or used <= threshold_mib:
            return
        time.sleep(1.0)


def current_gpu_memory_used_mib() -> int | None:
    command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    environment = payload["environment"]
    results = payload["results"]
    lines = [
        "# wllm Latency Suite",
        "",
        f"- Created UTC: {payload['created_utc']}",
        f"- Model: `{config['model']}`",
        f"- Profile: `{config['profile']}`",
        f"- Cases: {', '.join(f'`{name}`' for name in config['cases'])}",
        f"- Scenarios: {', '.join(f'`{name}`' for name in config['scenarios'])}",
        f"- Runs: {config['runs']} measured, {config['warmups']} warmup, batch size {config['batch_size']}",
        f"- Runtime: dtype={config['dtype']}, tp={config['tensor_parallel_size']}, "
        f"gpu_memory_utilization={config['gpu_memory_utilization']}, max_model_len={config['max_model_len']}",
        f"- Python: `{environment.get('python')}` via `{environment.get('executable')}`",
        f"- vLLM: `{environment.get('vllm_version')}`, torch: `{environment.get('torch_version')}`, "
        f"transformers: `{environment.get('transformers_version')}`",
        f"- GPU: `{environment.get('gpu')}`",
        "",
        "## Method",
        "",
        "Each case runs in a separate Python process so model instances do not share GPU memory. "
        "Warmups are excluded from steady-state statistics. Raw vLLM is the primary baseline per scenario; "
        "wllm completion measures the clean serving path; extraction cases include wllm trace timing metadata.",
        "",
    ]
    for scenario in config["scenarios"]:
        scenario_results = [result for result in results if result.get("scenario") == scenario]
        lines.extend(render_scenario_table(scenario, scenario_results))
    lines.extend(["", "## Case Descriptions", ""])
    for case_name in config["cases"]:
        lines.append(f"- `{case_name}`: {CASES[case_name].description}")
    lines.append("")
    return "\n".join(lines)


def render_scenario_table(scenario: str, results: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## Scenario `{scenario}`",
        "",
        "| Case | Median ms | P95 ms | Gen tok/s mean | Median vs raw vLLM | Load ms | Artifact bytes | "
        "Gen ms | Capture ms | Postprocess ms | Serialization ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if "error" in result:
            lines.append(
                f"| `{result.get('case')}` | ERROR: {escape_table(str(result.get('error')))} | | | | | | | | | |"
            )
            continue
        trace = result.get("summary", {}).get("trace_timing_ms", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result['case']}`",
                    fmt(nested(result, "summary", "wall_ms", "median")),
                    fmt(nested(result, "summary", "wall_ms", "p95")),
                    fmt(nested(result, "summary", "generated_tokens_per_s", "mean")),
                    fmt(result.get("median_wall_vs_raw_vllm")),
                    fmt(result.get("load_ms")),
                    fmt(nested(result, "summary", "artifact_bytes", "mean"), digits=0),
                    fmt(nested(trace, "generation", "median")),
                    fmt(nested(trace, "capture", "median")),
                    fmt(nested(trace, "postprocess", "median")),
                    fmt(nested(trace, "serialization", "median")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def nested(value: Any, *path: Any) -> Any:
    current = value
    for item in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(item)
        elif isinstance(current, list) and isinstance(item, int) and 0 <= item < len(current):
            current = current[item]
        else:
            return None
    return current


def fmt(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
