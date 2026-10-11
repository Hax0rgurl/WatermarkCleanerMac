# MAC WATERMARK CLEANER

A native Apple Silicon app powered by [DeMark-World](https://github.com/linkedlist771/DeMark-World), with automatic named-watermark scanning and editable timed regions.

[Download the latest Mac app](https://github.com/Hax0rgurl/WatermarkCleanerMac/releases/latest) · [Changes in 1.1.1](CHANGELOG.md) · [Verification and limitations](docs/VERIFICATION.md)

## What changed in 1.1.1

The earlier unrestricted detector could mistake white captions for watermarks. This app instead scans for the watermark names you specify, checks readable branding with Apple Vision OCR, and by default confirms the branding with a local vision model. It builds a timeline of separate appearances, rather than erasing a fixed box throughout the film.

- **Automatic scanning:** enter names such as `InVideo, Storyblocks, iStock`. Remove Watermarks scans automatically when no checked regions exist. Copied descriptions such as `:fish: InVideo/muse, S Storyblocks, iStock by Getty Images` are normalized to their brand names.
- **Actual video playback:** play, pause, scrub, step frames, jump between appearances, or play one selected interval.
- **Editable timed boxes:** start and stop at the playhead, adjust a detected box, or draw a missed watermark. Numeric calibration is optional. A newly drawn box initially covers two seconds, not the whole video.
- **Caption protection:** white text alone is insufficient. Automatic regions require matching named text, bounded boxes, and checked start/stop frames. Visual tracking uses references from confirmed branding; it does not treat arbitrary text as a new watermark.
- **Repeating marks:** one-second sampling discovers intermittent branding and reports its measured cadence. Boundary refinement checks included and neighboring excluded frames.
- **Local model fallback:** if an optional local vision server is unavailable, scanning uses the selected downloaded MLX model directly. Changing model settings preserves the checked timeline until an explicit rescan.
- **Reliable completion:** the app decodes the entire output, including audio, before marking the job complete. Failed or interrupted jobs are kept separate from finished videos.

The test film was 607.9 seconds long. The scan found 59 timed regions, including 43 repeating corner InVideo appearances with a measured median interval of 14.033 seconds. A subsequent full-length render completed and its entire video and audio decoded successfully. A separate short proof was visually checked for removal and preservation of two unrelated captions. These are specific test results, not a guarantee of perfect removal from every film.

## Install and use

Requires macOS 14 or later and Apple Silicon. The Quick vision profile is smaller; the Latest quality profile requires substantially more memory and is intended for Macs with at least 32 GB.

1. Download `MAC-WATERMARK-CLEANER-macOS-arm64-v1.1.1.zip` from the [latest release](https://github.com/Hax0rgurl/WatermarkCleanerMac/releases/latest), unzip it, and move **MAC WATERMARK CLEANER.app** into Applications.
2. Open the app and use **Choose Videos**. Initial runtime and model setup requires internet access.
3. Enter the watermark names, then use **Find & Check Watermarks** or **Remove Watermarks** to scan automatically.
4. Review the detected intervals with the scrubber and appearance navigation. Add or adjust any missed boxes at the playhead.
5. Remove the marks. Finished files are saved under `~/Movies/MAC WATERMARK CLEANER`.

The app has an ad-hoc signature and is not Apple-notarized. If Gatekeeper blocks it, use the macOS approval flow for opening a trusted downloaded application.

## Local vision models

The separate vision runtime reuses complete compatible model caches in the app, Hugging Face, or Pinokio. Missing weights are downloaded from pinned revisions:

| Profile | Model | Approximate first download |
|---|---|---|
| Quick | [Qwen3.5-4B, 4-bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) | 3 GB |
| Latest quality | [Qwen3.8-27B, 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 16 GB |

The model catalog changes with tested app releases; the app does not automatically execute whichever model happens to be newest online. Apple Vision OCR also supports scanning without the optional vision model. An already-running OpenAI-compatible vision server on this Mac can be selected using its loopback URL and model name. External vision servers and redirects are rejected; video frames stay local.

## Upstream contribution

[DeMark-World PR #11](https://github.com/linkedlist771/DeMark-World/pull/11) contributes an opt-in LaMa API for reviewed timed regions, protected compositing, streamed MP4 output, and lazy backend loading. It leaves the original detector workflow available. The full Mac scanner and native interface remain in this repository.

## Limits and file handling

Automatic scanning requires readable matching branding. It can miss unreadable symbols, brief appearances between samples, heavily obscured marks, or marks that move substantially. Manual timed regions remain available. LaMa reconstructs the selected pixels; the result depends on scene detail and mask size. Review the finished film before treating it as restored.

The native picker pins source-file identity. After complete output verification and identity rechecks, the original is moved to recoverable macOS Trash. Source-level CLI and Streamlit use leave originals in place. An uncertain Trash receipt is not reported as completion. Runtime data, models, caches, and logs live under `~/Library/Application Support/MAC WATERMARK CLEANER`, outside the signed app. Older app data paths and the historical bundle identifier remain compatible.

Use the tool with media you own or are authorized to modify.

## Build and verify

Requirements: macOS 14+, Apple Silicon, Xcode Command Line Tools, and `uv`.
The build script can use the `uv` copy installed with Pinokio. The app uses the
fully pinned and transitively resolved `requirements-macos.lock`; the verified
runtime is managed CPython 3.12.13.

```bash
./script/build_macos_app.sh
./script/build_and_run.sh --verify
```

Build outputs:

- `dist/MAC WATERMARK CLEANER.app`
- `dist/MAC-WATERMARK-CLEANER-macOS-arm64-v1.1.1.zip`
- `dist/MAC-WATERMARK-CLEANER-macOS-arm64-v1.1.1.zip.sha256`

To run the dedicated Mac/DeMark tests after the private runtime is provisioned:

```bash
MAC_WATERMARK_RUNTIME_DIR=/tmp/mac-watermark-cleaner-test-runtime \
  dist/MAC\ WATERMARK\ CLEANER.app/Contents/Resources/bootstrap_macos.sh test

dist/MAC\ WATERMARK\ CLEANER.app/Contents/Resources/tools/uv pip install \
  --python /tmp/mac-watermark-cleaner-test-runtime/Python/bin/python \
  -r requirements-macos-test.txt

/tmp/mac-watermark-cleaner-test-runtime/Python/bin/python -m pytest \
  tests/test_macos_branding.py tests/test_macos_single_window.py \
  tests/test_verified_trash.py tests/test_macos_upgrade.py
```

The historical CUDA/E2FGVI test directories are retained as upstream source;
they are not claimed as passing in the standard Mac runtime because MMCV/CUDA
is intentionally optional there.

## Source use

```bash
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock
PATH="$(.venv/bin/python -c 'import pathlib, imageio_ffmpeg; print(pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)'):$PATH" \
  .venv/bin/python -m streamlit run app.py
```

The CLI accepts one file or a folder:

```bash
.venv/bin/python cli.py -i input.mp4 -o output-cleaned.mp4
```

The source-level Streamlit and CLI entrypoints do not receive native Finder
URLs and therefore leave source files in place. Verified source-to-Trash
retirement is deliberately limited to the packaged Mac app's native picker.

## Licensing and attribution

The original SoraWatermarkCleaner source project remains under Apache License
2.0. The bundle identifier and legacy compatibility aliases retain that
historical name so existing app data and automation keep working; the product
and installed app name are **MAC WATERMARK CLEANER**.
The embedded DeMark-World engine is licensed by its actual repository license
file: MIT, copyright 2025 LLinkedlist. DeMark-World's README currently says
Apache-2.0, but its `LICENSE` file says MIT; this project preserves the MIT text
and attribution explicitly.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`THIRD_PARTY_LICENSES/` for full details.
