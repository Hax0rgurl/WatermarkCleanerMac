# MAC WATERMARK CLEANER

A local Apple Silicon video watermark cleaner with two restoration engines: **E2FGVI-HQ** and **ProPainter**. E2FGVI-HQ is selected by default. Both run on this Mac without paid credits or a video-upload service.

[Download the Mac app](https://github.com/Hax0rgurl/WatermarkCleanerMac/releases/latest) · [Changes](CHANGELOG.md) · [Verification](docs/VERIFICATION.md)

## Install and use

Requires **Apple Silicon and macOS 14 or later**.

1. Download `MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.1.zip`, unzip it, and move **MAC WATERMARK CLEANER.app** into Applications.
2. Open the app. The first launch automatically installs its private Python runtime and downloads **both** restoration model sets. Keep the internet connection available until setup finishes. The window shows setup and model-download progress.
3. Choose videos, select E2FGVI-HQ or ProPainter, and press **REMOVE WATERMARKS**.
4. Open the completed copy from the app or `~/Movies/MAC WATERMARK CLEANER`. Your original stays in place.

The four model checkpoints total about **364 MB**, in addition to Python and runtime dependencies. Later launches reuse verified local files. Interrupted setup can be retried; valid downloads are kept. The app uses tested, pinned model versions instead of choosing unverified new releases on each launch.

The app is ad-hoc signed and is not Apple-notarized. If macOS blocks the download, use its approval flow for opening a trusted downloaded application.

## Automatic detection and optional corrections

The default scan looks for a fixed list of common platform names. You can narrow it to specific branding in the optional watermark-name field, for example `InVideo, Storyblocks`.

Apple Vision reads text locally. Matching names seed a watermark track, then local pixel comparisons check its appearances and start/stop frames. The app does not use an unrestricted language-model detector or classify every white word as a watermark. Masks follow detected lettering and persistent logo pixels within the checked areas.

**Review or adjust watermark times** opens the source player. Play, pause, scrub, step frames, and set start or stop at the playhead. Draw a missed area or adjust an existing interval there. Numerical controls are optional calibration tools.

Readable named branding is required to seed automatic detection. Small, distorted, obscured, purely symbolic, or very brief marks can be missed. A brand appearing in unrelated on-screen text can still need correction. Review the result, especially detailed moving backgrounds and large marks. Neither model can guarantee a clean reconstruction of hidden scene detail.

## What changed in version 2

- Replaced the DeMark/LaMa restoration workflow with actual E2FGVI-HQ and ProPainter video models.
- Automatically prepares both model sets on first launch with checksum verification and retryable downloads.
- Uses adjacent video frames, bounded overlapping chunks, and shot boundaries during restoration.
- Preserves original frame timestamps, dimensions, and supported original audio, then decodes and verifies the completed file before publication.
- Keeps original videos in place. Existing output files are never overwritten.
- Removes the separate MLX/LLM vision setup, OpenCV runtime, and CUDA/MMCV requirement.

See [the desktop workflow](docs/interface.md) and [timed-region API](docs/timed-regions.md).

## Storage and privacy

Processing, text recognition, and model inference stay on the Mac. Setup downloads Python, dependencies, and published checkpoint files; it does not upload footage. Once setup is complete, the app can process locally without model downloads.

Runtime files, models, caches, selections, and logs live under:

```text
~/Library/Application Support/MAC WATERMARK CLEANER
```

The installed app contains its code and native tools. A failed or interrupted job is not offered as a completed output. Original videos are retained.

## Build and test

Developers need Xcode Command Line Tools and `uv`. The build script can also reuse the official `uv` executable bundled in an existing installed cleaner app.

```bash
./script/build_macos_app.sh
./script/build_and_run.sh --verify
```

The first command builds the app and validates its native executable, resources, signature, release ZIP, and checksum. The second also prepares the private runtime and models, checks local interface health, and opens the native app. Outputs are:

```text
dist/MAC WATERMARK CLEANER.app
dist/MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.1.zip
dist/MAC-WATERMARK-CLEANER-macOS-arm64-v2.0.1.zip.sha256
```

For source development and the full test suite:

```bash
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python -r requirements-macos-test.txt
./script/build_macos_app.sh
.venv/bin/ruff check app.py cli.py watermark_cleaner tests
.venv/bin/ruff format --check app.py cli.py watermark_cleaner tests
.venv/bin/python -m pytest
```

The suite includes actual pretrained CPU inference for **both** engines, so the first full test run downloads their verified checkpoints. See [tests/README.md](tests/README.md) for individual checks and caches. CI runs on an Apple Silicon macOS runner and keeps the native build and full test suite enabled.

To inspect the source interface after building the native OCR helper, first choose videos in the native app and close that app. The source interface can reuse its selection manifest:

```bash
MAC_WATERMARK_SELECTION_FILE="$HOME/Library/Application Support/MAC WATERMARK CLEANER/Selections/current.json" \
  .venv/bin/python -m streamlit run app.py
```

The native window provides the file picker; use the CLI below for standalone source processing. The interface prepares both model sets automatically. To prepare them without opening the interface:

```bash
.venv/bin/python -m watermark_cleaner.setup
.venv/bin/python -m watermark_cleaner.setup --status
```

## Command line

The CLI uses the same restoration and output-verification pipeline. It accepts one video or a folder of videos. Folder scans are nonrecursive and create separate timestamped MP4 copies.

```bash
.venv/bin/python -m watermark_cleaner.cli --help
.venv/bin/python -m watermark_cleaner.cli \
  -i input.mp4 -o cleaned.mp4 --engine e2fgvi --watermark-text InVideo
.venv/bin/python -m watermark_cleaner.cli \
  -i input.mp4 -o cleaned-propainter.mp4 --engine propainter --regions regions.json
```

Automatic scanning requires the built native text-recognition helper. Manual region JSON skips scanning. Source and CLI use retain originals, just like the native app. See [timed regions](docs/timed-regions.md) for the JSON format.

## Licenses and attribution

The app's original code retains its Apache-2.0 license. The bundled restoration backends have **noncommercial** terms: E2FGVI uses CC BY-NC 4.0, and ProPainter uses S-Lab License 1.0. The app license does not replace the backends' or checkpoints' upstream terms. RAFT and other retained components have their own notices.

Full texts and pinned source references are included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), `THIRD_PARTY_LICENSES/`, and the backend directories. This release does not bundle or invoke the old DeMark, LaMa, or Sora detector packages. Historical attribution is retained where required.
