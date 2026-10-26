#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"
PYTHON_VERSION="3.12.13"
RESOURCES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$RESOURCES_DIR/engine"
DEFAULT_RUNTIME_DIR="$HOME/Library/Application Support/MAC WATERMARK CLEANER"
RUNTIME_DIR="${MAC_WATERMARK_RUNTIME_DIR:-$DEFAULT_RUNTIME_DIR}"
VENV_DIR="$RUNTIME_DIR/Python"
BIN_DIR="$RUNTIME_DIR/bin"
LOG_DIR="$RUNTIME_DIR/Logs"
REQUIREMENTS="$ENGINE_DIR/requirements-macos.lock"
MARKER_FILE="$RUNTIME_DIR/runtime-requirements.sha256"
LOCK_DIR="$RUNTIME_DIR/.setup-lock"
setup_child=""
lock_owned=0

mkdir -p "$RUNTIME_DIR" "$BIN_DIR" "$LOG_DIR" "$RUNTIME_DIR/Cache" "$RUNTIME_DIR/Selections"
printf '\n[%s] Launcher session: %s\n' "$(/bin/date '+%Y-%m-%d %H:%M:%S')" "$MODE" >> "$LOG_DIR/launcher.log"
export MAC_WATERMARK_RUNTIME_DIR="$RUNTIME_DIR"
if [[ "$MODE" == "--test" || "$MODE" == "test" ]]; then
  export MAC_WATERMARK_OUTPUT_DIR="${MAC_WATERMARK_OUTPUT_DIR:-$RUNTIME_DIR/TestOutput}"
else
  export MAC_WATERMARK_OUTPUT_DIR="${MAC_WATERMARK_OUTPUT_DIR:-$HOME/Movies/MAC WATERMARK CLEANER}"
fi
export MAC_WATERMARK_SELECTION_FILE="${MAC_WATERMARK_SELECTION_FILE:-$RUNTIME_DIR/Selections/current.json}"
export UV_CACHE_DIR="$RUNTIME_DIR/Cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUNTIME_DIR/ManagedPython"
export UV_MANAGED_PYTHON=1
export XDG_CACHE_HOME="$RUNTIME_DIR/Cache"
export TORCH_HOME="$RUNTIME_DIR/Cache/torch"
export MPLCONFIGDIR="$RUNTIME_DIR/Cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ENGINE_DIR"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"

find_uv() {
  local candidate
  for candidate in "$RESOURCES_DIR/tools/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  command -v uv 2>/dev/null || return 1
}
UV_BIN="$(find_uv || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "ERROR: The app is missing its bundled setup tool. Download the complete app archive again."
  exit 69
fi
[[ -f "$ENGINE_DIR/app.py" && -f "$ENGINE_DIR/watermark_cleaner/setup.py" && -f "$REQUIREMENTS" ]] || {
  echo "ERROR: The app bundle is incomplete. Download the complete app archive again."
  exit 70
}
cd "$ENGINE_DIR"
if [[ "$MODE" == "--diagnose" || "$MODE" == "diagnose" ]]; then
  echo "DIAG_OK"
  echo "UV=$UV_BIN"
  echo "ENGINE=$ENGINE_DIR"
  echo "RUNTIME=$RUNTIME_DIR"
  echo "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  echo "OUTPUT=$MAC_WATERMARK_OUTPUT_DIR"
  exit 0
fi

release_lock() {
  if [[ "$lock_owned" == 1 && -f "$LOCK_DIR/pid" && "$(<"$LOCK_DIR/pid")" == "$$" ]]; then
    /bin/rm -f "$LOCK_DIR/pid"
    /bin/rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  lock_owned=0
}
cleanup() {
  if [[ -n "$setup_child" ]] && /bin/kill -0 "$setup_child" 2>/dev/null; then
    /bin/kill "$setup_child" 2>/dev/null || true
  fi
  release_lock
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

waited=0
while ! /bin/mkdir "$LOCK_DIR" 2>/dev/null; do
  owner=""
  if [[ -f "$LOCK_DIR/pid" ]]; then owner="$(<"$LOCK_DIR/pid")"; fi
  if [[ "$owner" =~ ^[0-9]+$ ]] && ! /bin/kill -0 "$owner" 2>/dev/null; then
    /bin/rm -f "$LOCK_DIR/pid"
    /bin/rmdir "$LOCK_DIR" 2>/dev/null || true
    continue
  fi
  if [[ -z "$owner" && "$waited" -gt 5 ]]; then
    /bin/rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  if [[ "$waited" -ge 900 ]]; then
    echo "ERROR: Another app window is still preparing the runtime. Close it and reopen the cleaner."
    exit 75
  fi
  if [[ "$waited" == 0 ]]; then echo "Waiting for the other setup window…"; fi
  /bin/sleep 1
  waited=$((waited + 1))
done
lock_owned=1
printf '%s\n' "$$" > "$LOCK_DIR/pid"

run_setup_child() {
  "$@" &
  setup_child=$!
  local result=0
  wait "$setup_child" || result=$?
  setup_child=""
  return "$result"
}

requirements_hash="$(/usr/bin/shasum -a 256 "$REQUIREMENTS" | /usr/bin/awk '{print $1}')"
installed_hash=""
if [[ -f "$MARKER_FILE" ]]; then installed_hash="$(<"$MARKER_FILE")"; fi
runtime_version=""
if [[ -x "$VENV_DIR/bin/python" ]]; then
  runtime_version="$("$VENV_DIR/bin/python" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
fi
if [[ ! -x "$VENV_DIR/bin/python" || "$runtime_version" != "$PYTHON_VERSION" || "$installed_hash" != "$requirements_hash" ]]; then
  echo "[1/5] Preparing private Python 3.12 runtime…"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    run_setup_child "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
  elif [[ "$runtime_version" != "$PYTHON_VERSION" ]]; then
    run_setup_child "$UV_BIN" venv --clear --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
  echo "[2/5] Installing local video components…"
  if ! run_setup_child "$UV_BIN" pip install --exact --require-hashes --only-binary :all: --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS"; then
    echo "ERROR: Component installation failed. Check the internet connection and reopen the app."
    exit 69
  fi
  "$VENV_DIR/bin/python" - <<'PY'
import av
import einops
import streamlit
import torch
import torchvision
from torchvision.ops import deform_conv2d
assert av.__version__ and einops.__version__ and streamlit.__version__
x = torch.ones(1, 1, 3, 3)
assert deform_conv2d(x, torch.zeros(1, 2, 3, 3), torch.ones(1, 1, 1, 1)).shape == x.shape
PY
  printf '%s\n' "$requirements_hash" > "$MARKER_FILE"
else
  echo "[1/5] Private Python runtime is ready."
  echo "[2/5] Local video components are current."
fi

echo "[3/5] Configuring video tools…"
ffmpeg_executable="$("$VENV_DIR/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$ffmpeg_executable" "$BIN_DIR/ffmpeg"
export PATH="$BIN_DIR:$VENV_DIR/bin:$PATH"
export MAC_WATERMARK_FFMPEG="$BIN_DIR/ffmpeg"
export MAC_WATERMARK_OCR_HELPER="${MAC_WATERMARK_OCR_HELPER:-$RESOURCES_DIR/tools/vision-ocr}"
export MAC_WATERMARK_TRASH_HELPER="${MAC_WATERMARK_TRASH_HELPER:-$RESOURCES_DIR/tools/trash-item}"

echo "[4/5] Preparing E2FGVI-HQ and ProPainter…"
if ! run_setup_child "$VENV_DIR/bin/python" -m watermark_cleaner.setup; then
  echo "ERROR: Model preparation did not finish. Reopen the app to retry; verified files are kept."
  exit 74
fi
release_lock
if [[ "$MODE" == "--prepare" || "$MODE" == "prepare" ]]; then
  echo "SETUP_READY"
  exit 0
fi

if [[ "$MODE" == "--test" || "$MODE" == "test" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
import app
import av
import sys
from watermark_cleaner.engines import EngineName
from watermark_cleaner.model_store import models_status
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert {engine.value for engine in EngineName} == {"e2fgvi", "propainter"}
assert all(models_status().values())
assert av.__version__ and callable(app.main)
assert "cv2" not in sys.modules
from importlib import metadata
assert not any(distribution.metadata["Name"].lower().startswith("opencv") for distribution in metadata.distributions())
print("RUNTIME_IMPORT_OK")
PY
  base_python="$("$VENV_DIR/bin/python" -c 'import pathlib, sys; print(pathlib.Path(getattr(sys, "_base_executable", sys.executable)).resolve())')"
  managed_python_root="$(cd "$UV_PYTHON_INSTALL_DIR" && /bin/pwd -P)"
  case "$base_python" in "$managed_python_root"/*) ;; *) echo "ERROR: Python is outside the private managed runtime: $base_python"; exit 72;; esac
  echo "MANAGED_PYTHON_OK=$base_python"
  test_port=18501
  while /usr/sbin/lsof -nP -iTCP:"$test_port" -sTCP:LISTEN >/dev/null 2>&1; do test_port=$((test_port + 1)); done
  "$VENV_DIR/bin/python" -m streamlit run "$ENGINE_DIR/app.py" --server.address 127.0.0.1 --server.port "$test_port" --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false > "$LOG_DIR/streamlit-health-test.log" 2>&1 &
  health_pid=$!
  health_ok=0
  for _ in {1..80}; do
    if /usr/bin/curl -fsS "http://127.0.0.1:$test_port/_stcore/health" >/dev/null 2>&1; then health_ok=1; break; fi
    /bin/sleep 0.25
  done
  /bin/kill "$health_pid" >/dev/null 2>&1 || true
  wait "$health_pid" >/dev/null 2>&1 || true
  if [[ "$health_ok" != 1 ]]; then
    echo "ERROR: The local interface health check failed."
    /usr/bin/tail -n 40 "$LOG_DIR/streamlit-health-test.log" || true
    exit 73
  fi
  echo "STREAMLIT_HEALTH_OK=http://127.0.0.1:$test_port/_stcore/health"
  exit 0
fi

echo "[5/5] Starting MAC WATERMARK CLEANER…"
port=8501
while [[ "$port" -le 8510 ]]; do
  if ! /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then break; fi
  port=$((port + 1))
done
if [[ "$port" -gt 8510 ]]; then echo "ERROR: Ports 8501–8510 are already in use."; exit 71; fi
echo "MAC_WATERMARK_UI_URL=http://127.0.0.1:$port"
exec "$VENV_DIR/bin/python" -m streamlit run "$ENGINE_DIR/app.py" --server.address 127.0.0.1 --server.port "$port" --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false --theme.base dark
