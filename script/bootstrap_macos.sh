#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"
PYTHON_VERSION="3.12.13"
RESOURCES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$RESOURCES_DIR/engine"
DEFAULT_RUNTIME_DIR="$HOME/Library/Application Support/MAC WATERMARK CLEANER"
LEGACY_RUNTIME_DIR="$HOME/Library/Application Support/Sora Watermark Cleaner"
if [[ -z "${MAC_WATERMARK_RUNTIME_DIR:-}" \
  && -z "${SORA_WATERMARK_RUNTIME_DIR:-}" \
  && ! -e "$DEFAULT_RUNTIME_DIR" \
  && -d "$LEGACY_RUNTIME_DIR" ]]; then
  if /bin/ln -s "$LEGACY_RUNTIME_DIR" "$DEFAULT_RUNTIME_DIR"; then
    echo "Reusing existing app data through $DEFAULT_RUNTIME_DIR"
  else
    echo "WARNING: Existing app data could not be linked; using it in place."
    DEFAULT_RUNTIME_DIR="$LEGACY_RUNTIME_DIR"
  fi
fi
RUNTIME_DIR="${MAC_WATERMARK_RUNTIME_DIR:-${SORA_WATERMARK_RUNTIME_DIR:-$DEFAULT_RUNTIME_DIR}}"
VENV_DIR="$RUNTIME_DIR/Python"
BIN_DIR="$RUNTIME_DIR/bin"
LOG_DIR="$RUNTIME_DIR/Logs"
LOG_FILE="$LOG_DIR/launcher.log"
MARKER_FILE="$RUNTIME_DIR/runtime-requirements.sha256"
REQUIREMENTS="$ENGINE_DIR/requirements-macos.lock"

mkdir -p "$RUNTIME_DIR" "$BIN_DIR" "$LOG_DIR" "$RUNTIME_DIR/Cache"
mkdir -p "$RUNTIME_DIR/Selections"
touch "$LOG_FILE"
printf '\n[%s] Launcher session: %s\n' \
  "$(/bin/date '+%Y-%m-%d %H:%M:%S')" "$MODE" >> "$LOG_FILE"

export MAC_WATERMARK_RUNTIME_DIR="$RUNTIME_DIR"
export SORA_WATERMARK_RUNTIME_DIR="$RUNTIME_DIR"
if [[ "$MODE" == "--test" || "$MODE" == "test" ]]; then
  OUTPUT_DIR="${MAC_WATERMARK_OUTPUT_DIR:-${SORA_WATERMARK_OUTPUT_DIR:-$RUNTIME_DIR/TestOutput}}"
else
  OUTPUT_DIR="${MAC_WATERMARK_OUTPUT_DIR:-${SORA_WATERMARK_OUTPUT_DIR:-$HOME/Movies/MAC WATERMARK CLEANER}}"
fi
export MAC_WATERMARK_OUTPUT_DIR="$OUTPUT_DIR"
export SORA_WATERMARK_OUTPUT_DIR="$OUTPUT_DIR"
export MAC_WATERMARK_SELECTION_FILE="${MAC_WATERMARK_SELECTION_FILE:-$RUNTIME_DIR/Selections/current.json}"
export UV_CACHE_DIR="$RUNTIME_DIR/Cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUNTIME_DIR/ManagedPython"
export UV_MANAGED_PYTHON=1
export XDG_CACHE_HOME="$RUNTIME_DIR/Cache"
export TORCH_HOME="$RUNTIME_DIR/Cache/torch"
export HF_HOME="$RUNTIME_DIR/Cache/huggingface"
export MPLCONFIGDIR="$RUNTIME_DIR/Cache/matplotlib"
export YOLO_CONFIG_DIR="$RUNTIME_DIR/Cache/ultralytics"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ENGINE_DIR"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
mkdir -p \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$MPLCONFIGDIR" \
  "$YOLO_CONFIG_DIR"
cd "$ENGINE_DIR"

find_uv() {
  local candidate
  for candidate in \
    "$RESOURCES_DIR/tools/uv" \
    "$HOME/pinokio/bin/miniforge/bin/uv" \
    "/opt/homebrew/bin/uv" \
    "/usr/local/bin/uv"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  command -v uv 2>/dev/null || return 1
}

UV_BIN="$(find_uv || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "ERROR: uv is required to create the private Python runtime."
  echo "Install uv from https://docs.astral.sh/uv/ and reopen the app."
  exit 69
fi

if [[ "$MODE" == "--diagnose" || "$MODE" == "diagnose" ]]; then
  [[ -f "$ENGINE_DIR/app.py" ]] || { echo "ERROR: engine/app.py is missing"; exit 70; }
  [[ -f "$REQUIREMENTS" ]] || { echo "ERROR: requirements-macos.lock is missing"; exit 70; }
  echo "DIAG_OK"
  echo "UV=$UV_BIN"
  echo "ENGINE=$ENGINE_DIR"
  echo "RUNTIME=$RUNTIME_DIR"
  echo "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  echo "OUTPUT=$MAC_WATERMARK_OUTPUT_DIR"
  exit 0
fi

requirements_hash="$(/usr/bin/shasum -a 256 "$REQUIREMENTS" | /usr/bin/awk '{print $1}')"
installed_hash=""
if [[ -f "$MARKER_FILE" ]]; then
  installed_hash="$(<"$MARKER_FILE")"
fi
runtime_python_version=""
if [[ -x "$VENV_DIR/bin/python" ]]; then
  runtime_python_version="$("$VENV_DIR/bin/python" -c \
    'import platform; print(platform.python_version())' 2>/dev/null || true)"
fi

if [[ ! -x "$VENV_DIR/bin/python" \
  || "$runtime_python_version" != "$PYTHON_VERSION" \
  || "$installed_hash" != "$requirements_hash" ]]; then
  echo "[1/4] Preparing private Python 3.12 runtime…"
  if ! /usr/bin/curl -IfsS --connect-timeout 5 https://pypi.org/simple/ >/dev/null; then
    echo "WARNING: Network check failed. Cached packages will still be tried."
    echo "First-time setup normally requires an internet connection."
  fi
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
  elif [[ "$runtime_python_version" != "$PYTHON_VERSION" ]]; then
    "$UV_BIN" venv --clear --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
  echo "[2/4] Installing or updating app components…"
  if ! "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS"; then
    echo "ERROR: Runtime installation failed."
    echo "Check the internet connection, then reopen the app. No original videos were touched."
    exit 69
  fi
  printf '%s\n' "$requirements_hash" > "$MARKER_FILE"
else
  echo "[1/4] Private Python runtime is ready."
  echo "[2/4] App components are current."
fi

echo "[3/4] Configuring bundled video tools…"
ffmpeg_executable="$("$VENV_DIR/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$ffmpeg_executable" "$BIN_DIR/ffmpeg"
export PATH="$BIN_DIR:$VENV_DIR/bin:$PATH"
export MAC_WATERMARK_FFMPEG="$BIN_DIR/ffmpeg"
export MAC_WATERMARK_TRASH_HELPER="${MAC_WATERMARK_TRASH_HELPER:-$RESOURCES_DIR/tools/trash-item}"

if [[ "$MODE" == "--test" || "$MODE" == "test" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
import app
import ffmpeg
import sys
import torch
import ultralytics
from demark_world import core
from demark_world import completion
from demark_world.configs import OUTPUT_DIR, RUNTIME_ROOT
from demark_world.schemas import CleanerType
import cv2
from importlib import metadata
assert CleanerType.LAMA.value == "lama"
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert str(OUTPUT_DIR)
assert str(RUNTIME_ROOT)
opencv_distributions = [
    name
    for name in (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    )
    if any(
        dist.metadata["Name"].lower() == name
        for dist in metadata.distributions()
    )
]
assert opencv_distributions == ["opencv-python"], opencv_distributions
assert cv2.__version__
assert torch.__version__
assert ultralytics.__version__
assert ffmpeg.input
assert core.DeMarkWorld
assert completion.finalize_cleaned_video
assert app.main
print("RUNTIME_IMPORT_OK")
PY
  base_python="$("$VENV_DIR/bin/python" -c \
    'import pathlib, sys; print(pathlib.Path(getattr(sys, "_base_executable", sys.executable)).resolve())')"
  managed_python_root="$(cd "$UV_PYTHON_INSTALL_DIR" && /bin/pwd -P)"
  case "$base_python" in
    "$managed_python_root"/*) ;;
    *)
      echo "ERROR: Managed Python escaped the private runtime: $base_python"
      exit 72
      ;;
  esac
  echo "MANAGED_PYTHON_OK=$base_python"
  "$BIN_DIR/ffmpeg" -version | /usr/bin/head -n 1
  test_port=18501
  while /usr/sbin/lsof -nP -iTCP:"$test_port" -sTCP:LISTEN >/dev/null 2>&1; do
    test_port=$((test_port + 1))
  done
  "$VENV_DIR/bin/python" -m streamlit run "$ENGINE_DIR/app.py" \
    --server.address 127.0.0.1 \
    --server.port "$test_port" \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false \
    > "$LOG_DIR/streamlit-health-test.log" 2>&1 &
  health_pid=$!
  health_ok=0
  for _ in {1..80}; do
    if /usr/bin/curl -fsS \
      "http://127.0.0.1:$test_port/_stcore/health" >/dev/null 2>&1; then
      health_ok=1
      break
    fi
    /bin/sleep 0.25
  done
  /bin/kill "$health_pid" >/dev/null 2>&1 || true
  wait "$health_pid" >/dev/null 2>&1 || true
  if [[ "$health_ok" -ne 1 ]]; then
    echo "ERROR: Streamlit health check failed."
    /usr/bin/tail -n 40 "$LOG_DIR/streamlit-health-test.log" || true
    exit 73
  fi
  echo "STREAMLIT_HEALTH_OK=http://127.0.0.1:$test_port/_stcore/health"
  exit 0
fi

echo "[4/4] Starting MAC WATERMARK CLEANER…"
port=8501
while [[ "$port" -le 8510 ]]; do
  if ! /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  port=$((port + 1))
done
if [[ "$port" -gt 8510 ]]; then
  echo "ERROR: Ports 8501–8510 are already in use."
  exit 71
fi

echo "MAC_WATERMARK_UI_URL=http://127.0.0.1:$port"
exec "$VENV_DIR/bin/python" -m streamlit run "$ENGINE_DIR/app.py" \
  --server.address 127.0.0.1 \
  --server.port "$port" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
