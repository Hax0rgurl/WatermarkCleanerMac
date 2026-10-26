#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="MAC WATERMARK CLEANER"
APP_BUNDLE="$ROOT_DIR/dist/$APP_NAME.app"
APP_BINARY="$APP_BUNDLE/Contents/MacOS/$APP_NAME"
RUNTIME_LOG="$HOME/Library/Application Support/MAC WATERMARK CLEANER/Logs/launcher.log"

pkill -x "$APP_NAME" >/dev/null 2>&1 || true
"$ROOT_DIR/script/build_macos_app.sh"

case "$MODE" in
  run)
    /usr/bin/open -n "$APP_BUNDLE"
    ;;
  --debug|debug)
    /usr/bin/lldb -- "$APP_BINARY"
    ;;
  --logs|logs|--telemetry|telemetry)
    /usr/bin/open -n "$APP_BUNDLE"
    mkdir -p "$(dirname "$RUNTIME_LOG")"
    touch "$RUNTIME_LOG"
    /usr/bin/tail -f "$RUNTIME_LOG"
    ;;
  --verify|verify)
    "$APP_BINARY" --self-test
    diagnostic_runtime="$(mktemp -d "${TMPDIR:-/tmp}/mac-watermark-cleaner-verify.XXXXXX")"
    trap 'rm -rf "$diagnostic_runtime"' EXIT
    MAC_WATERMARK_RUNTIME_DIR="$diagnostic_runtime" \
      /bin/bash "$APP_BUNDLE/Contents/Resources/bootstrap_macos.sh" --diagnose
    /bin/bash "$APP_BUNDLE/Contents/Resources/bootstrap_macos.sh" --test
    /usr/bin/open -n "$APP_BUNDLE"
    for _ in {1..40}; do
      if /usr/bin/pgrep -x "$APP_NAME" >/dev/null; then
        echo "NATIVE_APP_PROCESS_OK"
        exit 0
      fi
      /bin/sleep 0.25
    done
    echo "ERROR: The native application did not stay running." >&2
    exit 74
    ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify]" >&2
    exit 2
    ;;
esac
