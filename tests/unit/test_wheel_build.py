from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

EXPECTED_FLAT_PACKAGES = [
    "artifacts",
    "extractors",
    "research",
    "runtime",
    "schemas",
    "server",
    "tracing",
]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build the wheel once for the module; isolated build fetches hatchling."""
    outdir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"wheel build failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    wheels = list(outdir.glob("*.whl"))
    assert wheels, f"no wheel produced in {outdir}"
    return wheels[0]


def test_wheel_contains_all_flat_layout_packages(built_wheel) -> None:
    """VAL-CROSS-024: wheel includes every flat-layout package and cli.py at root."""
    names = zipfile.ZipFile(built_wheel).namelist()

    # cli.py must be at the wheel root
    assert "cli.py" in names, "cli.py missing from wheel root"

    for pkg in EXPECTED_FLAT_PACKAGES:
        assert any(n.startswith(pkg + "/") for n in names), f"missing flat package: {pkg}/"

    # No src/wllm/ directory, no nested wllm package, no src/ prefix at all
    assert not any(n.startswith("src/")
                   for n in names), f"src/ paths leaked into wheel: {[n for n in names if n.startswith('src/')]}"
    assert not any(n.startswith("wllm/") for n in names), "nested wllm/ package found in wheel"

    # Wheel metadata present
    assert any(n.endswith(".dist-info/METADATA") for n in names), "missing METADATA"


def test_wheel_entry_point_resolves(built_wheel, tmp_path) -> None:
    """VAL-CROSS-025: installed wheel exposes `wllm` console script that resolves
    to cli.main, and `wllm --help` does not import vllm or torch."""
    venv_dir = tmp_path / "entrypoint_venv"
    create = subprocess.run(
        [str(VENV_PYTHON), "-m", "venv", str(venv_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert create.returncode == 0, f"venv creation failed: {create.stderr}"

    venv_python = venv_dir / "bin" / "python"
    venv_wllm = venv_dir / "bin" / "wllm"

    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", str(built_wheel)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (
        f"pip install --no-deps failed:\nstdout:\n{install.stdout}\nstderr:\n{install.stderr}"
    )

    # The console script must exist and `wllm --help` must exit 0 with expected text
    help_run = subprocess.run(
        [str(venv_wllm), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_run.returncode == 0, f"wllm --help failed:\n{help_run.stderr}"
    assert "wllm" in help_run.stdout
    assert "serve" in help_run.stdout
    assert "doctor" in help_run.stdout

    doctor_help = subprocess.run(
        [str(venv_wllm), "doctor", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert doctor_help.returncode == 0, f"wllm doctor --help failed:\n{doctor_help.stderr}"
    assert "Check the local wllm/vLLM environment" in doctor_help.stdout

    # Running cli.main(['--help']) from the installed wheel must not import vllm/torch
    probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import sys; "
                "import cli; "
                "rc = cli.main(['--help']); "
                "print('imported_vllm=' + str('vllm' in sys.modules)); "
                "print('imported_torch=' + str('torch' in sys.modules)); "
                "raise SystemExit(rc)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, f"probe failed:\n{probe.stderr}"
    assert "imported_vllm=False" in probe.stdout
    assert "imported_torch=False" in probe.stdout


def test_wheel_entry_point_metadata_declares_cli_main(built_wheel) -> None:
    """The wheel's entry_points.txt must declare `wllm = cli:main`."""
    names = zipfile.ZipFile(built_wheel).namelist()
    entry_path = next(n for n in names if n.endswith("entry_points.txt"))
    text = zipfile.ZipFile(built_wheel).read(entry_path).decode()
    assert "console_scripts" in text
    assert "wllm = cli:main" in text
