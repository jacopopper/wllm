from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import pytest

import cli


def test_help_does_not_import_vllm(capsys) -> None:
    sys.modules.pop("vllm", None)
    exit_code = cli.main(["--help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "serve" in captured.out
    assert "vllm" not in sys.modules


def test_help_does_not_import_torch(capsys) -> None:
    sys.modules.pop("torch", None)
    exit_code = cli.main(["--help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "serve" in captured.out
    assert "torch" not in sys.modules


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


def test_serve_command_rejects_missing_model() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])
