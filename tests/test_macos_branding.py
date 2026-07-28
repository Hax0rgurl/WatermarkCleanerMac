from __future__ import annotations

import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_NAME = "MAC WATERMARK CLEANER"
VERSION = "1.0.4"


def test_product_name_is_consistent_in_user_facing_source() -> None:
    required_text = {
        "app.py": [
            f'page_title="{PRODUCT_NAME}"',
            f"<h1>🎬 {PRODUCT_NAME}</h1>",
        ],
        "macos/Launcher.swift": [
            f'private let appName = "{PRODUCT_NAME}"',
            "MAC_WATERMARK_UI_URL=",
        ],
        "script/bootstrap_macos.sh": [
            f"Application Support/{PRODUCT_NAME}",
            f"Movies/{PRODUCT_NAME}",
            f"Starting {PRODUCT_NAME}",
        ],
        "script/build_macos_app.sh": [
            f'APP_NAME="{PRODUCT_NAME}"',
            f'VERSION="{VERSION}"',
            "MAC-WATERMARK-CLEANER-macOS-arm64",
        ],
    }

    for relative_path, snippets in required_text.items():
        content = (ROOT / relative_path).read_text()
        for snippet in snippets:
            assert snippet in content, f"{relative_path} is missing {snippet!r}"


def test_built_bundle_branding_when_bundle_exists() -> None:
    bundle = ROOT / "dist" / f"{PRODUCT_NAME}.app"
    if not bundle.exists():
        return

    with (bundle / "Contents" / "Info.plist").open("rb") as handle:
        info = plistlib.load(handle)

    assert info["CFBundleDisplayName"] == PRODUCT_NAME
    assert info["CFBundleName"] == PRODUCT_NAME
    assert info["CFBundleExecutable"] == PRODUCT_NAME
    assert info["CFBundleShortVersionString"] == VERSION
    assert (bundle / "Contents" / "MacOS" / PRODUCT_NAME).is_file()
