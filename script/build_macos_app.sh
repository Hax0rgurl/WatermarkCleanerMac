#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="MAC WATERMARK CLEANER"
VERSION="2.0.0"
IDENTIFIER="com.hax0rgurl.MacWatermarkCleaner"
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
CHECKSUM_PATH="$ZIP_PATH.sha256"
SWIFTC="${SWIFTC:-}"
UV_SOURCE="${UV_SOURCE:-$(command -v uv || true)}"
if [[ -z "$UV_SOURCE" ]]; then
  for candidate in \
    "/Applications/$APP_NAME.app/Contents/Resources/tools/uv" \
    "$HOME/Applications/$APP_NAME.app/Contents/Resources/tools/uv"; do
    if [[ -x "$candidate" ]]; then
      UV_SOURCE="$candidate"
      break
    fi
  done
fi
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

rm -rf "$BUILD_DIR" "$FINAL_APP_BUNDLE" "$ZIP_PATH" "$CHECKSUM_PATH"
mkdir -p "$BUILD_DIR" "$MACOS_DIR" "$ENGINE_DIR" "$RESOURCES_DIR/tools" "$DIST_DIR"
mkdir -p "$BUILD_DIR/ModuleCache"
mkdir -p "$STAGING_ROOT/native"
# Compile stable local copies. File Provider can change hydrated source-file
# metadata while Swift is reading an iCloud-backed workspace.
cp "$ROOT_DIR/macos/Launcher.swift" "$ROOT_DIR/macos/TrashHelper.swift" \
  "$ROOT_DIR/macos/VisionOCR.swift" "$STAGING_ROOT/native/"

echo "Compiling native launcher…"
swift_args=(
  -O
  -target arm64-apple-macos14.0
  -module-cache-path "$BUILD_DIR/ModuleCache"
  -framework AppKit
  -framework Foundation
  -framework UniformTypeIdentifiers
  -framework WebKit
  -o "$MACOS_DIR/$APP_NAME"
  "$STAGING_ROOT/native/Launcher.swift"
)
if [[ -n "$SWIFTC" ]]; then
  "$SWIFTC" "${swift_args[@]}"
else
  /usr/bin/xcrun swiftc "${swift_args[@]}"
fi

echo "Compiling recoverable Trash helper…"
trash_helper_args=(
  -O
  -target arm64-apple-macos14.0
  -module-cache-path "$BUILD_DIR/ModuleCache"
  -framework Foundation
  -o "$RESOURCES_DIR/tools/trash-item"
  "$STAGING_ROOT/native/TrashHelper.swift"
)
if [[ -n "$SWIFTC" ]]; then
  "$SWIFTC" "${trash_helper_args[@]}"
else
  /usr/bin/xcrun swiftc "${trash_helper_args[@]}"
fi

echo "Compiling local watermark text checker…"
ocr_helper_args=(
  -O
  -target arm64-apple-macos14.0
  -module-cache-path "$BUILD_DIR/ModuleCache"
  -framework Foundation
  -framework Vision
  -o "$RESOURCES_DIR/tools/vision-ocr"
  "$STAGING_ROOT/native/VisionOCR.swift"
)
if [[ -n "$SWIFTC" ]]; then
  "$SWIFTC" "${ocr_helper_args[@]}"
else
  /usr/bin/xcrun swiftc "${ocr_helper_args[@]}"
fi
cp "$RESOURCES_DIR/tools/vision-ocr" "$BUILD_DIR/vision-ocr"

echo "Staging immutable engine source…"
rsync -a \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT_DIR/watermark_cleaner/" "$ENGINE_DIR/watermark_cleaner/"
cp "$ROOT_DIR/app.py" "$ENGINE_DIR/app.py"
cp "$ROOT_DIR/requirements-macos.txt" "$ENGINE_DIR/requirements-macos.txt"
cp "$ROOT_DIR/requirements-macos.lock" "$ENGINE_DIR/requirements-macos.lock"

cp "$ROOT_DIR/script/bootstrap_macos.sh" "$RESOURCES_DIR/bootstrap_macos.sh"
chmod +x "$RESOURCES_DIR/bootstrap_macos.sh"
cp "$ROOT_DIR/LICENSE" "$RESOURCES_DIR/LICENSE-Apache-2.0.txt"
cp "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$RESOURCES_DIR/THIRD_PARTY_NOTICES.md"
cp -R "$ROOT_DIR/THIRD_PARTY_LICENSES" "$RESOURCES_DIR/THIRD_PARTY_LICENSES"

if [[ -x "$UV_SOURCE" ]]; then
  cp "$UV_SOURCE" "$RESOURCES_DIR/tools/uv"
  chmod +x "$RESOURCES_DIR/tools/uv"
else
  echo "ERROR: UV_SOURCE must point to the official uv executable before packaging." >&2
  exit 69
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
    "CFBundleVersion": "200",
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
/usr/bin/codesign --force --sign - "$RESOURCES_DIR/tools/trash-item"
/usr/bin/codesign --force --sign - "$RESOURCES_DIR/tools/vision-ocr"
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
(
  cd "$DIST_DIR"
  /usr/bin/shasum -a 256 "$(basename "$ZIP_PATH")" \
    > "$(basename "$CHECKSUM_PATH")"
  /usr/bin/shasum -a 256 -c "$(basename "$CHECKSUM_PATH")"
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
# File Provider may immediately restore metadata on an app copied into an
# iCloud-backed workspace. The fresh-unzipped release above is the canonical
# distribution validation; retry the local convenience copy narrowly and
# report workspace-only metadata without invalidating the already-proven ZIP.
/usr/bin/xattr -d com.apple.FinderInfo "$FINAL_APP_BUNDLE" 2>/dev/null || true
/usr/bin/xattr -d com.apple.fileprovider.fpfs#P "$FINAL_APP_BUNDLE" 2>/dev/null || true
final_validation_log="$BUILD_DIR/final-codesign.log"
if ! /usr/bin/codesign --verify --deep --strict --verbose=2 \
  "$FINAL_APP_BUNDLE" 2> "$final_validation_log"; then
  if /usr/bin/grep -q \
    'resource fork, Finder information, or similar detritus not allowed' \
    "$final_validation_log"; then
    echo "WARNING: The workspace File Provider restored Finder metadata on the local app copy."
    echo "The canonical ZIP passed strict signing after a fresh extraction outside the workspace."
  else
    /bin/cat "$final_validation_log" >&2
    exit 1
  fi
fi

echo "APP=$FINAL_APP_BUNDLE"
echo "ZIP=$ZIP_PATH"
echo "CHECKSUM=$CHECKSUM_PATH"
