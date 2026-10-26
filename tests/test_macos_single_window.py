from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_embeds_local_cleaner_in_one_webview_window() -> None:
    launcher = (ROOT / "macos/Launcher.swift").read_text(encoding="utf-8")

    assert "import WebKit" in launcher
    assert "WKWebView" in launcher
    assert "webView.load(" in launcher
    assert "NSOpenPanel" in launcher
    assert "MAC_WATERMARK_SELECTION_FILE" in launcher
    assert "MAC_WATERMARK_TRASH_HELPER" in launcher
    assert "struct SourceRecord" in launcher
    assert "file_type" in launcher
    assert "mtime_ns" in launcher
    assert "lstat(" in launcher
    assert "NSWorkspace.shared.open(uiURL)" not in launcher
    assert "diagnosticsView.isHidden = true" in launcher


def test_native_picker_paths_replace_byte_only_streamlit_uploads() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "_load_native_selection()" in app
    assert "validate_source_identity(" in app
    assert "move_to_macos_trash(" not in app
    assert "st.file_uploader" not in app
    assert "SourceIdentity.from_mapping(" in app
    assert "remains exactly where it was" not in app
    assert "Those originals were not moved" not in app


def test_recoverable_trash_uses_foundation_not_permanent_unlink() -> None:
    helper = (ROOT / "macos/TrashHelper.swift").read_text(encoding="utf-8")

    assert "FileManager.default.trashItem" in helper
    assert "lstat(" in helper
    assert 'status: "location_unknown"' in helper
    assert "current == expected" in helper
    assert ".removeItem" not in helper
    assert "unlink" not in helper


def test_startup_waits_for_verified_models_before_the_interface() -> None:
    bootstrap = (ROOT / "script/bootstrap_macos.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "macos/Launcher.swift").read_text(encoding="utf-8")
    assert bootstrap.index("-m watermark_cleaner.setup") < bootstrap.index("MAC_WATERMARK_UI_URL=")
    assert "--require-hashes" in bootstrap
    assert "pendingBackendOutput" in launcher
    assert "MAC_WATERMARK_MODEL_PROGRESS=" in launcher
    assert "keeps your originals" in launcher
    assert "DeMark-World engine" not in launcher


def test_vaporwave_family_tokens_are_present_and_readable() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8").lower()
    launcher = (ROOT / "macos/Launcher.swift").read_text(encoding="utf-8")

    for color in ("#ff006e", "#00f5ff", "#8338ec", "#39ff14"):
        assert color in app
    assert "repeating-linear-gradient" in app
    assert "background-size: 24px 24px" in app
    assert "monospacedSystemFont" in launcher
    assert "Palette.cyan" in launcher
    assert "Palette.pink" in launcher


def test_release_checksum_is_portable_and_zip_rejects_appledouble() -> None:
    build_script = (ROOT / "script/build_macos_app.sh").read_text(encoding="utf-8")

    assert 'cd "$DIST_DIR"' in build_script
    assert 'shasum -a 256 "$(basename "$ZIP_PATH")"' in build_script
    assert "__MACOSX" in build_script
    assert "\\._" in build_script
