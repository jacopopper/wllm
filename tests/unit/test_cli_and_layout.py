from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

import pytest

import cli


def test_help_does_not_import_vllm() -> None:
    result = _help_import_probe("vllm")
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "imported=False" in result.stdout


def test_help_does_not_import_torch() -> None:
    result = _help_import_probe("torch")
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "imported=False" in result.stdout


def _help_import_probe(module_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules.pop(%r, None); "
                "import cli; "
                "rc = cli.main(['--help']); "
                "print('imported=' + str(%r in sys.modules)); "
                "raise SystemExit(rc)"
            )
            % (module_name, module_name),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_serve_command_parses_local_files_only() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--local-files-only"])
    assert args.model == "Qwen/Qwen3-0.6B"
    assert args.local_files_only is True


def test_no_src_wllm_package() -> None:
    assert not Path("src/wllm").exists()


def test_pytest_import_path_includes_project_root() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'pythonpath = ["src", "."]' in pyproject


def test_vllm_extra_targets_tested_vllm_line() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'vllm = ["vllm==0.10.2"]' in pyproject


def test_benchmark_smoke_help_runs_from_project_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Compare trace-free" in result.stdout


def test_serve_command_parses_common_vllm_options() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen3-0.6B",
            "--host", "0.0.0.0",
            "--port", "8100",
            "--dtype", "bfloat16",
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "4096",
            "--tokenizer", "custom/tokenizer",
            "--served-model-name", "my-model",
            "--api-key", "secret-key",
            "--seed", "42",
            "--trust-remote-code",
            "--local-files-only",
            "--prewarm-hidden-states",
            "--enable-online-hidden-states",
        ]
    )
    assert args.model == "Qwen/Qwen3-0.6B"
    assert args.host == "0.0.0.0"
    assert args.port == 8100
    assert args.dtype == "bfloat16"
    assert args.tensor_parallel_size == 2
    assert args.gpu_memory_utilization == 0.85
    assert args.max_model_len == 4096
    assert args.tokenizer == "custom/tokenizer"
    assert args.served_model_name == "my-model"
    assert args.api_key == "secret-key"
    assert args.seed == 42
    assert args.trust_remote_code is True
    assert args.local_files_only is True
    assert args.prewarm_hidden_states is True
    assert args.enable_online_hidden_states is True


def test_serve_command_rejects_missing_model() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])


# ---------------------------------------------------------------------------
# VAL-CLI-001 / VAL-CLI-002: help text (top-level and serve) imports nothing heavy
# ---------------------------------------------------------------------------


def _import_probe(module_name: str, cli_argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run cli.main(cli_argv) in a subprocess and report heavy-module import status."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules.pop('vllm', None); "
                "sys.modules.pop('torch', None); "
                "import cli; "
                "rc = cli.main(%r); "
                "print('imported_vllm=' + str('vllm' in sys.modules)); "
                "print('imported_torch=' + str('torch' in sys.modules)); "
                "raise SystemExit(rc)"
            )
            % (cli_argv,),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_serve_help_does_not_import_vllm_or_torch() -> None:
    result = _import_probe("vllm", ["serve", "--help"])
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "imported_vllm=False" in result.stdout
    assert "imported_torch=False" in result.stdout


# ---------------------------------------------------------------------------
# VAL-CLI-004 .. VAL-CLI-021: accepted options, defaults and explicit values
# ---------------------------------------------------------------------------


def test_accepted_option_defaults() -> None:
    """Every accepted option has the documented default when omitted."""
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B"])
    assert args.host == "127.0.0.1"  # VAL-CLI-004
    assert args.port == 8000 and isinstance(args.port, int)  # VAL-CLI-005
    assert args.dtype == "auto"  # VAL-CLI-006
    assert args.tensor_parallel_size == 1 and isinstance(args.tensor_parallel_size, int)  # VAL-CLI-007
    assert args.gpu_memory_utilization == 0.9 and isinstance(args.gpu_memory_utilization, float)  # VAL-CLI-008
    assert args.max_model_len is None  # VAL-CLI-009
    assert args.tokenizer is None  # VAL-CLI-010
    assert args.served_model_name is None  # VAL-CLI-011
    assert args.api_key is None  # VAL-CLI-012
    assert args.seed is None  # VAL-CLI-013
    assert args.trust_remote_code is False  # VAL-CLI-014
    assert args.local_files_only is False  # VAL-CLI-015
    assert args.prewarm_hidden_states is False  # VAL-CLI-017
    assert args.enable_online_hidden_states is False  # VAL-CLI-018
    assert args.artifact_dir == "./wllm-artifacts"  # VAL-CLI-019
    assert args.log_level == "info"  # VAL-CLI-021


def test_accepted_option_explicit_values() -> None:
    """Every accepted option reflects its explicit value, with correct types."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve", "Qwen/Qwen3-0.6B",
            "--host", "0.0.0.0",
            "--port", "8100",
            "--dtype", "float16",
            "--tensor-parallel-size", "4",
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "4096",
            "--tokenizer", "my/tokenizer",
            "--served-model-name", "my-model",
            "--api-key", "sk-abc123",
            "--seed", "42",
            "--trust-remote-code",
            "--local-files-only",
            "--prewarm-hidden-states",
            "--enable-online-hidden-states",
            "--artifact-dir", "/tmp/artifacts",
            "--log-level", "debug",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 8100 and isinstance(args.port, int)
    assert args.dtype == "float16"
    assert args.tensor_parallel_size == 4 and isinstance(args.tensor_parallel_size, int)
    assert args.gpu_memory_utilization == 0.5 and isinstance(args.gpu_memory_utilization, float)
    assert args.max_model_len == 4096 and isinstance(args.max_model_len, int)
    assert args.tokenizer == "my/tokenizer"
    assert args.served_model_name == "my-model"
    assert args.api_key == "sk-abc123"
    assert args.seed == 42 and isinstance(args.seed, int)
    assert args.trust_remote_code is True
    assert args.local_files_only is True
    assert args.prewarm_hidden_states is True
    assert args.enable_online_hidden_states is True
    assert args.artifact_dir == "/tmp/artifacts"
    assert args.log_level == "debug"


def test_port_rejects_non_integer() -> None:
    """VAL-CLI-005: argparse rejects non-integer --port values."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--port", "abc"])


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
def test_log_level_accepts_valid_choices(level: str) -> None:
    """VAL-CLI-020: the four documented log-level values are accepted."""
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--log-level", level])
    assert args.log_level == level


@pytest.mark.parametrize("level", ["trace", "critical", "fatal", "verbose"])
def test_log_level_rejects_invalid_choices(level: str) -> None:
    """VAL-CLI-020: values outside the choices list are rejected by argparse."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--log-level", level])


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "Warning", "ERROR"])
def test_log_level_is_case_sensitive(level: str) -> None:
    """VAL-CLI-036: --log-level choices match case-sensitively; uppercase rejected."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--log-level", level])


# ---------------------------------------------------------------------------
# VAL-CLI-022: every accepted option appears in `wllm serve --help`
# ---------------------------------------------------------------------------


def _serve_subparser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    subparsers_action = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    return subparsers_action.choices["serve"]


def test_serve_help_lists_every_accepted_option() -> None:
    help_text = _serve_subparser().format_help()
    expected_flags = [
        "--host",
        "--port",
        "--dtype",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--tokenizer",
        "--served-model-name",
        "--api-key",
        "--seed",
        "--trust-remote-code",
        "--local-files-only",
        "--prewarm-hidden-states",
        "--enable-online-hidden-states",
        "--artifact-dir",
        "--log-level",
    ]
    for flag in expected_flags:
        assert flag in help_text, f"missing flag in serve --help: {flag}"
    # positional model argument
    assert "model" in help_text


# ---------------------------------------------------------------------------
# VAL-CLI-023 .. VAL-CLI-032: unsupported vLLM flags are rejected by argparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag_argv",
    [
        ["--quantization", "awq"],
        ["--kv-cache-dtype", "fp8"],
        ["--device", "cpu"],
        ["--enforce-eager"],
        ["--disable-log-requests"],
        ["--swap-space", "4"],
        ["--max-num-seqs", "256"],
        ["--max-num-batched-tokens", "2048"],
        ["--download-dir", "/tmp/models"],
        ["--pipeline-parallel-size", "2"],
    ],
)
def test_unsupported_vllm_flags_rejected(flag_argv: list[str]) -> None:
    """Each unsupported vLLM flag must be rejected by argparse (SystemExit)."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", *flag_argv])


# ---------------------------------------------------------------------------
# VAL-CLI-033: every parsed option maps to a config field or named consumer
# ---------------------------------------------------------------------------


def _run_cmd_serve_with_fakes(args: argparse.Namespace, captured: dict, monkeypatch) -> int:
    """Run cli._cmd_serve with heavy deps replaced by fakes that record their inputs."""
    import artifacts.store as artifacts_store_mod
    import runtime.vllm_runtime as vllm_runtime_mod
    import server.app as server_app_mod
    import uvicorn

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def prewarm_hidden_states(self):
            captured["prewarmed"] = True

    class FakeStore:
        def __init__(self, path):
            captured["artifact_dir"] = path

    def fake_create_app(runtime, artifact_store, api_key):
        captured["api_key"] = api_key
        captured["runtime"] = runtime
        return object()

    def fake_uvicorn_run(app, host, port, log_level):
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntime", FakeRuntime)
    monkeypatch.setattr(vllm_runtime_mod, "VLLMRuntimeConfig", vllm_runtime_mod.VLLMRuntimeConfig)
    monkeypatch.setattr(artifacts_store_mod, "ArtifactStore", FakeStore)
    monkeypatch.setattr(server_app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    return int(cli._cmd_serve(args) or 0)


def test_cmd_serve_propagates_every_parsed_option(monkeypatch, tmp_path) -> None:
    """VAL-CLI-033: no accepted option is silently dropped before reaching its consumer."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve", "Qwen/Qwen3-0.6B",
            "--host", "0.0.0.0",
            "--port", "8100",
            "--dtype", "bfloat16",
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "4096",
            "--tokenizer", "custom/tok",
            "--served-model-name", "my-model",
            "--api-key", "sk-abc",
            "--seed", "42",
            "--trust-remote-code",
            "--local-files-only",
            "--prewarm-hidden-states",
            "--enable-online-hidden-states",
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--log-level", "debug",
        ]
    )
    captured: dict = {}
    rc = _run_cmd_serve_with_fakes(args, captured, monkeypatch)
    assert rc == 0

    config = captured["config"]
    assert config.model == "Qwen/Qwen3-0.6B"
    assert config.served_model_name == "my-model"
    assert config.dtype == "bfloat16"
    assert config.tensor_parallel_size == 2
    assert config.gpu_memory_utilization == 0.5
    assert config.max_model_len == 4096
    assert config.tokenizer == "custom/tok"
    assert config.seed == 42
    assert config.trust_remote_code is True
    assert config.local_files_only is True
    assert config.prewarm_hidden_states is True
    assert config.enable_online_hidden_states is True
    # Non-config args consumed by their named consumers
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8100
    assert captured["api_key"] == "sk-abc"
    assert captured["artifact_dir"] == tmp_path / "artifacts"
    assert captured["log_level"] == "debug"
    assert captured.get("prewarmed") is True

    # Every argparse attribute (except command/func) must be consumed by either
    # the VLLMRuntimeConfig fields or the named non-config consumers.
    config_fields = set(vars(config))
    consumed_elsewhere = {"host", "port", "api_key", "artifact_dir", "log_level"}
    parsed_attrs = set(vars(args)) - {"command", "func"}
    leftover = parsed_attrs - config_fields - consumed_elsewhere
    assert not leftover, f"parsed but dropped before reaching a consumer: {leftover}"


# ---------------------------------------------------------------------------
# VAL-CLI-016: --local-files-only propagates to runtime config
# ---------------------------------------------------------------------------


def test_local_files_only_propagates_to_runtime_config(monkeypatch, tmp_path) -> None:
    parser = cli.build_parser()
    args_with = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--local-files-only", "--artifact-dir", str(tmp_path)])
    captured: dict = {}
    _run_cmd_serve_with_fakes(args_with, captured, monkeypatch)
    assert captured["config"].local_files_only is True

    args_without = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--artifact-dir", str(tmp_path)])
    captured_without: dict = {}
    _run_cmd_serve_with_fakes(args_without, captured_without, monkeypatch)
    assert captured_without["config"].local_files_only is False


# ---------------------------------------------------------------------------
# VAL-CLI-034: model argument accepts HF IDs, absolute paths, relative paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-0.6B", "/data/models/llama", "./local-model", "../sibling/model"],
)
def test_model_argument_accepts_hf_ids_and_paths(model: str) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["serve", model])
    assert args.model == model


def test_model_argument_rejects_empty_string() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", ""])


# ---------------------------------------------------------------------------
# VAL-CLI-035: --local-files-only produces structured startup error when model absent
# ---------------------------------------------------------------------------


def test_local_files_only_startup_error_when_model_absent(monkeypatch) -> None:
    """When --local-files-only is set and the model is absent, vLLM init fails
    with a structured RuntimeUnavailableError referencing --local-files-only."""
    import runtime.vllm_runtime as vllm_runtime_mod
    from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
    from server.errors import RuntimeUnavailableError

    class AbsentModelLLM:
        def __init__(self, **kwargs):
            raise FileNotFoundError(f"model not found: {kwargs.get('model')}")

    class Imports:
        module = object()
        version = "0.10.2"
        LLM = AbsentModelLLM
        SamplingParams = type("SamplingParams", (), {"__init__": lambda self, **k: None})

    monkeypatch.setattr(vllm_runtime_mod, "import_vllm", lambda: Imports())
    runtime = VLLMRuntime(VLLMRuntimeConfig(model="/nonexistent/missing-model", local_files_only=True))

    with pytest.raises(RuntimeUnavailableError) as excinfo:
        runtime._ensure_loaded()

    err = excinfo.value
    assert err.code == "vllm_initialization_failed"
    assert err.status_code == 503
    assert err.details.get("local_files_only") is True
    assert "--local-files-only" in err.message
    # raw vLLM traceback must not leak as the message
    assert "Traceback" not in err.message


# ---------------------------------------------------------------------------
# VAL-CLI-037: bare `wllm` invocation prints help and exits 0 without heavy imports
# ---------------------------------------------------------------------------


def test_bare_wllm_prints_help_and_exits_zero() -> None:
    result = _import_probe("vllm", [])
    assert result.returncode == 0
    # Help text is printed to stdout by parser.print_help()
    assert "wllm" in result.stdout
    assert "serve" in result.stdout
    assert "imported_vllm=False" in result.stdout
    assert "imported_torch=False" in result.stdout


def test_bare_wllm_main_returns_zero_in_process(capsys) -> None:
    """Direct in-process call: main([]) returns 0 and prints usage."""
    rc = cli.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "wllm" in captured.out
    assert "serve" in captured.out


# ---------------------------------------------------------------------------
# VAL-CLI-003: missing model argument fails with error mentioning model
# ---------------------------------------------------------------------------


def test_serve_without_model_mentions_model_in_error(capsys) -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["serve"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "model" in captured.err.lower() or "required" in captured.err.lower()


# ---------------------------------------------------------------------------
# Edge-case validation: negative port, non-positive tensor-parallel-size,
# out-of-range gpu-memory-utilization, non-positive max-model-len
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_port", ["-1", "0", "65536", "99999", "-100"])
def test_port_rejects_out_of_range(bad_port: str) -> None:
    """--port rejects negative, zero, and >65535 values."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--port", bad_port])


@pytest.mark.parametrize("bad_tp", ["0", "-1", "-8"])
def test_tensor_parallel_size_rejects_non_positive(bad_tp: str) -> None:
    """--tensor-parallel-size rejects zero and negative values."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--tensor-parallel-size", bad_tp])


@pytest.mark.parametrize("bad_gmu", ["-0.1", "1.1", "2.0", "-1.0"])
def test_gpu_memory_utilization_rejects_out_of_range(bad_gmu: str) -> None:
    """--gpu-memory-utilization rejects values outside [0.0, 1.0]."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--gpu-memory-utilization", bad_gmu])


@pytest.mark.parametrize("bad_ml", ["0", "-1", "-128"])
def test_max_model_len_rejects_non_positive(bad_ml: str) -> None:
    """--max-model-len rejects zero and negative values."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--max-model-len", bad_ml])


def test_port_accepts_boundary_values() -> None:
    """--port accepts 1 and 65535 (boundary values)."""
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--port", "1"])
    assert args.port == 1
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--port", "65535"])
    assert args.port == 65535


def test_gpu_memory_utilization_accepts_boundary_values() -> None:
    """--gpu-memory-utilization accepts 0.0 and 1.0 (boundary values)."""
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--gpu-memory-utilization", "0.0"])
    assert args.gpu_memory_utilization == 0.0
    args = parser.parse_args(["serve", "Qwen/Qwen3-0.6B", "--gpu-memory-utilization", "1.0"])
    assert args.gpu_memory_utilization == 1.0
