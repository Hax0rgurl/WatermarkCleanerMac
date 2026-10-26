from __future__ import annotations

import plistlib
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_NAME = "MAC WATERMARK CLEANER"
VERSION = "2.0.1"


def test_package_metadata_matches_app_release_version() -> None:
    from watermark_cleaner import __version__

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == VERSION == __version__
    assert '"CFBundleVersion": "201"' in (ROOT / "script/build_macos_app.sh").read_text()


def test_product_name_is_consistent_in_user_facing_source() -> None:
    required_text = {
        "app.py": [
            f'page_title="{PRODUCT_NAME}"',
            f"<h1>{PRODUCT_NAME}</h1>",
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
    assert info["CFBundleIdentifier"] == "com.hax0rgurl.MacWatermarkCleaner"
    assert not (bundle / "Contents/Resources/engine/demark_world").exists()
    assert (bundle / "Contents/Resources/engine/watermark_cleaner/engines.py").is_file()
