# Tests

Run these commands from the repository root on Apple Silicon with macOS 14 or later. Xcode Command Line Tools build the native launcher and OCR helper.

```bash
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python -r requirements-macos-test.txt
./script/build_macos_app.sh
.venv/bin/ruff check app.py cli.py watermark_cleaner tests
.venv/bin/ruff format --check app.py cli.py watermark_cleaner tests
.venv/bin/python -m pytest
```

The full suite includes real, pretrained CPU inference for E2FGVI-HQ and ProPainter. Their first test run downloads four checksum-verified checkpoint files, about 364 MB total. These tests do not upload media. Synthetic inputs test channel order, dimensions, masks, temporal inference, and the CPU deformable-convolution adapter.

`WATERMARK_TEST_MODEL_DIR` can point to a complete verified checkpoint cache with `e2fgvi/` and `propainter/` subdirectories. Without it, backend tests use the system temporary directory's `watermark-cleaner-pretrained-test-cache`. A corrupted or incomplete file is downloaded again.

Useful individual checks:

```bash
.venv/bin/python -m pytest tests/test_temporal_engines.py
.venv/bin/python -m pytest tests/test_model_store.py tests/test_cli.py
.venv/bin/python -m pytest tests/test_macos_upgrade.py tests/test_macos_single_window.py tests/test_macos_branding.py
.venv/bin/python -m pytest tests/test_app_workflow.py
```

The remaining tests cover source identity, timing validation, scanner scope, masks, streaming playback, processing failures, output verification, and native packaging. Some pipeline tests inject a small restoration function to exercise failure paths without model inference; the separate real-checkpoint tests keep backend inference covered.

To exercise actual first-launch setup in a separate runtime without opening a window:

```bash
MAC_WATERMARK_RUNTIME_DIR=/private/tmp/mac-watermark-cleaner-first-launch \
  /bin/bash 'dist/MAC WATERMARK CLEANER.app/Contents/Resources/bootstrap_macos.sh' --prepare

MAC_WATERMARK_RUNTIME_DIR=/private/tmp/mac-watermark-cleaner-first-launch \
  /bin/bash 'dist/MAC WATERMARK CLEANER.app/Contents/Resources/bootstrap_macos.sh' --test
```

`--prepare` installs the private runtime and verifies both model sets. `--test` additionally checks imports, the private managed Python path, absence of OpenCV, and local Streamlit health. These are startup checks, not visual proof of a restored video. `--diagnose` checks bundle layout without downloading models.

The GitHub workflow runs the authored-source checks, builds and verifies the native app, runs the complete suite including both real backends, checks imports from the packaged copy, and saves the release archive and test report. No model or native-build check is disabled to obtain a passing result.
