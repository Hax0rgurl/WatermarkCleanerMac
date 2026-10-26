"""First-launch and cached-launch contracts for the two-engine desktop app."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from watermark_cleaner import setup

ROOT = Path(__file__).resolve().parents[1]


def test_model_setup_prepares_both_engines_and_reports_progress(monkeypatch, capsys) -> None:
    calls = []

    def prepare(engine_id, progress=None):
        calls.append(engine_id)
        progress(0.5, "Downloading a local model")

    monkeypatch.setattr(setup, "ensure_models", prepare)
    monkeypatch.setattr(setup, "models_status", lambda: {"e2fgvi": True, "propainter": True})
    assert setup.main([]) == 0
    assert calls == [None]
    output = capsys.readouterr().out
    assert "MAC_WATERMARK_MODEL_PROGRESS=0.5000|Downloading a local model" in output
    assert '"e2fgvi": true' in output and '"propainter": true' in output


def test_model_setup_does_not_claim_readiness_after_partial_download(monkeypatch, capsys) -> None:
    monkeypatch.setattr(setup, "ensure_models", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "models_status", lambda: {"e2fgvi": True, "propainter": False})
    assert setup.main([]) == 1
    assert "Both restoration models must be ready" in capsys.readouterr().err


def test_model_setup_status_never_downloads(monkeypatch) -> None:
    def unexpected_download(*args, **kwargs):
        raise AssertionError("The status command must stay offline")

    monkeypatch.setattr(setup, "ensure_models", unexpected_download)
    monkeypatch.setattr(setup, "models_status", lambda: {"e2fgvi": True, "propainter": True})
    assert setup.main(["--status"]) == 0


def test_model_setup_download_failure_is_nonzero_and_actionable(monkeypatch, capsys) -> None:
    def failed_download(*args, **kwargs):
        raise OSError("connection interrupted")

    monkeypatch.setattr(setup, "ensure_models", failed_download)
    assert setup.main([]) == 1
    error = capsys.readouterr().err
    assert "connection interrupted" in error and "Verified downloads will be reused" in error


@pytest.fixture
def staged_bootstrap(tmp_path: Path):
    resources = tmp_path / "Resources"
    engine = resources / "engine"
    package = engine / "watermark_cleaner"
    package.mkdir(parents=True)
    (engine / "app.py").write_text("# staged UI\n")
    (package / "setup.py").write_text("# staged model setup\n")
    lock = engine / "requirements-macos.lock"
    lock.write_text("# test runtime lock\n")
    script = resources / "bootstrap_macos.sh"
    script.write_bytes((ROOT / "script/bootstrap_macos.sh").read_bytes())
    tools = resources / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("#!/bin/bash\nexit 99\n")
    uv.chmod(0o755)
    runtime = tmp_path / "Runtime"
    python = runtime / "Python/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/bash\n"
        'if [[ "${1:-}" == \'-c\' && "${2:-}" == *platform.python_version* ]]; then\n'
        "  echo 3.12.13\n"
        'elif [[ "${1:-}" == \'-c\' && "${2:-}" == *imageio_ffmpeg* ]]; then\n'
        "  echo /usr/bin/true\n"
        "elif [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'watermark_cleaner.setup' ]]; then\n"
        "  echo MODEL_PREPARATION_CALLED\n"
        '  exit "${MODEL_SETUP_RESULT:-0}"\n'
        "else\n"
        "  echo UNEXPECTED_RUNTIME_COMMAND >&2\n"
        "  exit 98\n"
        "fi\n"
    )
    python.chmod(0o755)
    (runtime / "runtime-requirements.sha256").write_text(
        hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    environment = os.environ | {"MAC_WATERMARK_RUNTIME_DIR": str(runtime)}
    return script, runtime, environment


def test_cached_launch_still_checks_both_model_sets(staged_bootstrap) -> None:
    script, runtime, environment = staged_bootstrap
    result = subprocess.run(
        ["/bin/bash", str(script), "--prepare"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Local video components are current" in result.stdout
    assert "MODEL_PREPARATION_CALLED" in result.stdout
    assert "SETUP_READY" in result.stdout
    assert "MAC_WATERMARK_UI_URL=" not in result.stdout
    assert not (runtime / ".setup-lock").exists()


def test_failed_model_setup_never_starts_interface(staged_bootstrap) -> None:
    script, runtime, environment = staged_bootstrap
    result = subprocess.run(
        ["/bin/bash", str(script), "serve"],
        env=environment | {"MODEL_SETUP_RESULT": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 74
    assert "Model preparation did not finish" in result.stdout
    assert "MAC_WATERMARK_UI_URL=" not in result.stdout
    assert not (runtime / ".setup-lock").exists()


def test_bundle_diagnostics_check_new_package_before_setup(staged_bootstrap) -> None:
    script, runtime, environment = staged_bootstrap
    result = subprocess.run(
        ["/bin/bash", str(script), "--diagnose"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DIAG_OK" in result.stdout and "MODEL_PREPARATION_CALLED" not in result.stdout
    (script.parent / "engine/watermark_cleaner/setup.py").unlink()
    result = subprocess.run(
        ["/bin/bash", str(script), "--diagnose"], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 70
    assert "bundle is incomplete" in result.stdout
