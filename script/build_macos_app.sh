#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="MAC WATERMARK CLEANER"
VERSION="1.0.4"
IDENTIFIER="com.hax0rgurl.SoraWatermarkCleaner"
BUILD_DIR="$ROOT_DIR/.build/macos"
DIST_DIR="$ROOT_DIR/dist"
FINAL_APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
STAGING_ROOT="$(mktemp -d "${TMPDIR:-/private/tmp}/mac-watermark-cleaner-package.XXXXXX")"
APP_BUNDLE="$STAGING_ROOT/$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"
ENGINE_DIR="$RESOURCES_DIR/engine"
ZIP_PATH="$DIST_DIR/MAC-WATERMARK-CLEANER-macOS-arm64-v$VERSION.zip"
SWIFTC="${SWIFTC:-}"
UV_SOURCE="${UV_SOURCE:-$HOME/pinokio/bin/miniforge/bin/uv}"
diagnostic_runtime=""
extract_dir=""

cleanup() {
  rm -rf "$STAGING_ROOT"
  if [[ -n "$diagnostic_runtime" ]]; then
    rm -rf "$diagnostic_runtime"
  fi
  if [[ -n "$extract_dir" ]]; then
    rm -rf "$extract_dir"
  fi
}
trap cleanup EXIT
export COPYFILE_DISABLE=1

rm -rf "$BUILD_DIR" "$FINAL_APP_BUNDLE" "$ZIP_PATH"
mkdir -p "$BUILD_DIR" "$MACOS_DIR" "$ENGINE_DIR" "$RESOURCES_DIR/tools" "$DIST_DIR"
mkdir -p "$BUILD_DIR/ModuleCache"

echo "Compiling native launcher…"
swift_args=(
  -O
  -target arm64-apple-macos14.0
  -module-cache-path "$BUILD_DIR/ModuleCache"
  -framework AppKit
  -framework Foundation
  -o "$MACOS_DIR/$APP_NAME"
  "$ROOT_DIR/macos/Launcher.swift"
)
if [[ -n "$SWIFTC" ]]; then
  "$SWIFTC" "${swift_args[@]}"
else
  /usr/bin/xcrun swiftc "${swift_args[@]}"
fi

echo "Staging immutable engine source…"
rsync -a \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'iopaint/tests' \
  "$ROOT_DIR/demark_world/" "$ENGINE_DIR/demark_world/"
cp "$ROOT_DIR/app.py" "$ENGINE_DIR/app.py"
cp "$ROOT_DIR/requirements-macos.txt" "$ENGINE_DIR/requirements-macos.txt"
cp "$ROOT_DIR/requirements-macos.lock" "$ENGINE_DIR/requirements-macos.lock"
mkdir -p "$ENGINE_DIR/resources"
if [[ -f "$ROOT_DIR/resources/watermark_template.png" ]]; then
  cp "$ROOT_DIR/resources/watermark_template.png" "$ENGINE_DIR/resources/"
fi

cp "$ROOT_DIR/script/bootstrap_macos.sh" "$RESOURCES_DIR/bootstrap_macos.sh"
chmod +x "$RESOURCES_DIR/bootstrap_macos.sh"
cp "$ROOT_DIR/LICENSE" "$RESOURCES_DIR/LICENSE-Apache-2.0.txt"
cp "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$RESOURCES_DIR/THIRD_PARTY_NOTICES.md"
cp -R "$ROOT_DIR/THIRD_PARTY_LICENSES" "$RESOURCES_DIR/THIRD_PARTY_LICENSES"

if [[ -x "$UV_SOURCE" ]]; then
  cp "$UV_SOURCE" "$RESOURCES_DIR/tools/uv"
  chmod +x "$RESOURCES_DIR/tools/uv"
else
  echo "WARNING: uv was not bundled; the app will look for an installed copy."
fi

echo "Building icon…"
cp "$ROOT_DIR/packaging/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"

/usr/bin/python3 - "$CONTENTS/Info.plist" "$APP_NAME" "$IDENTIFIER" "$VERSION" <<'PY'
import plistlib
import sys

path, app_name, identifier, version = sys.argv[1:]
payload = {
    "CFBundleDevelopmentRegion": "en",
    "CFBundleDisplayName": app_name,
    "CFBundleExecutable": app_name,
    "CFBundleIconFile": "AppIcon",
    "CFBundleIdentifier": identifier,
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": app_name,
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": version,
    "CFBundleVersion": "104",
    "LSMinimumSystemVersion": "14.0",
    "NSHighResolutionCapable": True,
    "NSPrincipalClass": "NSApplication",
}
with open(path, "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=True)
PY

echo "Signing and validating bundle…"
/usr/bin/xattr -cr "$APP_BUNDLE"
if [[ -x "$RESOURCES_DIR/tools/uv" ]]; then
  /usr/bin/codesign --force --sign - "$RESOURCES_DIR/tools/uv"
fi
/usr/bin/codesign --force --sign - "$MACOS_DIR/$APP_NAME"
/usr/bin/codesign --force --deep --sign - "$APP_BUNDLE"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
/usr/bin/plutil -lint "$CONTENTS/Info.plist"
/usr/bin/vtool -show-build "$MACOS_DIR/$APP_NAME"
if ! /usr/bin/vtool -show-build "$MACOS_DIR/$APP_NAME" | \
  /usr/bin/grep -Eq 'minos[[:space:]]+14\.0'; then
  echo "ERROR: Launcher does not declare the required macOS 14 minimum." >&2
  exit 1
fi
"$MACOS_DIR/$APP_NAME" --self-test

diagnostic_runtime="$(mktemp -d "${TMPDIR:-/tmp}/mac-watermark-cleaner-diagnose.XXXXXX")"
MAC_WATERMARK_RUNTIME_DIR="$diagnostic_runtime" \
  /bin/bash "$RESOURCES_DIR/bootstrap_macos.sh" --diagnose

(
  cd "$STAGING_ROOT"
  /usr/bin/zip -q -X -r -y "$ZIP_PATH" "$APP_NAME.app"
)
/usr/bin/unzip -t "$ZIP_PATH" >/dev/null
if /usr/bin/unzip -Z1 "$ZIP_PATH" | \
  /usr/bin/grep -Eq '(^|/)__MACOSX/|(^|/)\._'; then
  echo "ERROR: ZIP contains forbidden AppleDouble metadata." >&2
  exit 1
fi

extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/mac-watermark-cleaner-extract.XXXXXX")"
/usr/bin/unzip -q "$ZIP_PATH" -d "$extract_dir"
extracted_app="$extract_dir/$APP_NAME.app"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$extracted_app"
"$extracted_app/Contents/MacOS/$APP_NAME" --self-test

/usr/bin/ditto --norsrc --noextattr --noacl \
  "$APP_BUNDLE" "$FINAL_APP_BUNDLE"
/usr/bin/xattr -cr "$FINAL_APP_BUNDLE"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$FINAL_APP_BUNDLE"

echo "APP=$FINAL_APP_BUNDLE"
echo "ZIP=$ZIP_PATH"
