# Version 2 verification

This document distinguishes startup/package checks from a complete media-quality review. Commands for reproducing the suite are in [tests/README.md](../tests/README.md).

## Startup and packaging checks executed

On October 26, 2026, an empty, isolated runtime successfully:

- Downloaded managed CPython 3.12.13.
- Installed the fully hashed production dependency lock.
- Downloaded and verified all four E2FGVI-HQ and ProPainter checkpoint files.
- Reported both model sets ready.
- Reopened using the cached runtime and models without installing dependencies or downloading checkpoint files again.

The production runtime imported PyAV 19.0.2, Torch 2.13.0, and Torchvision 0.28.0. It had no OpenCV distribution and did not import `cv2`. Its base Python was inside the private managed runtime.

The native version 2 app build passed strict signature verification, resource self-tests, macOS 14 minimum checks, bootstrap diagnostics, ZIP integrity checks, SHA-256 verification, and self-tests after a fresh ZIP extraction. The packaged bootstrap then passed its runtime import and local Streamlit health checks against that isolated runtime.

These checks prove setup and package integrity. They do not by themselves prove that every restored frame looks correct, that the installed Applications copy matches a future release, or that a remote CI run has completed.

The new CLI also completed a real four-frame E2FGVI-HQ restoration using explicit timed regions. Independent checks confirmed 96×64 dimensions, all four frames, exact timestamps, and an unchanged source SHA-256. This synthetic smoke test checks the command path; it is not a full-film visual assessment. The built Python wheel's installed command printed the expected version and help from outside the checkout.

## Media and regression coverage

The complete local suite passed: **167 tests**, including actual pretrained inference for both engines, native Apple Vision text recognition, native recoverable Trash receipts, CLI execution, real MP4 encoding/muxing/full decoding, and output-publication races. Authored Python lint and format checks also passed.

The supplied 372-frame variable-rate sample was scanned locally. The scan found the TikTok watermark in frames 0 through 127 at its first position and frames 128 through 371 at its second position. These bounds were checked against decoded source frames. Mask inspection prompted a tighter adjoining-logo search to exclude a bright scene detail.

The full suite exercises both real pretrained CPU backends, timed masks, invalid input, local scanning, exact video timestamps, audio preservation, output publication, source identity, CLI behavior, and restoration failure paths. Some failure-path tests inject a restoration function; real-checkpoint tests remain separate and enabled.

A release is published only after the full macOS workflow succeeds. Its release job uses the same run's app ZIP and checksum. An existing version tag pointing to another commit is rejected, and an existing published release is never replaced.

## Practical limits

Automatic detection starts from readable matching platform names, then compares local image detail across frames. It can miss tiny, unreadable, symbolic, brief, or unusual moving watermarks. Optional visual corrections remain available. A result's decodability does not establish perfect reconstruction of every hidden background detail.

The output contract is H.264, 8-bit 4:2:0 MP4 using even dimensions, the first video stream, exact decoded frame timestamps, and supported original audio streams. Encoding is lossy. HDR, transparency, subtitles, and additional video streams are outside this contract. Unsupported MP4 audio or failed verification stops publication and preserves the source.

Version 1 verification records and old DeMark/LaMa operational instructions were removed from the current documentation. Their historical source remains in Git history; those results are not presented as proof of version 2 behavior.
