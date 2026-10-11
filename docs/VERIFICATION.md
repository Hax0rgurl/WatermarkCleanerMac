# Version 1.1.1 verification

## Success criteria

- Installed app, release archive, and published engine source match.
- Relevant macOS, restoration, timed-region, branding, playback, and completion tests pass.
- Release signature, native self-test, and archive extraction are checked.
- Published commit and downloaded release asset match local records.
- Upstream contribution is reviewed against the original repository.
- Obsolete local versions are removed only after publication verification.

## Executed media checks

A 607.9-second, 1280×720, 30 fps film was scanned locally using Qwen3.5-4B, Apple Vision OCR, and model-confirmed visual tracking. It yielded 59 checked intervals and 234 boundary checks. Its 43 repeating corner InVideo appearances had a median cycle of 14.033 seconds. Other intervals covered Storyblocks, iStock/Getty branding, and an InVideo ending card.

No selected region was active on the previously affected caption frames at 5:00 and 8:00. A separate 16-second proof used the Latest quality profile with an unavailable localhost server. Direct Qwen3.8-27B fallback detected and timed the InVideo mark; LaMa removed it. All 480 frames and audio decoded, and visual inspection confirmed the two tested captions remained intact.

The subsequent full-length app render completed at 03:11 ET on October 10, 2026. The completion receipt confirmed full-output verification. An independent FFmpeg decode with `-xerror` also succeeded for all video and audio. This verifies completion and decodability; every restored frame has not been visually graded.

## Limits

A separate app crash was not reproduced. Successful rendering is not proof that every possible crash is eliminated. Automatic detection does not guarantee coverage of unreadable logos, short appearances between samples, or arbitrary moving watermarks. Review the timeline and finished film. CUDA/E2FGVI and the historical vendored test suites are not claimed as passing on this Mac.

## Publication checks

All 99 relevant tests passed on the verified CPython 3.12.13 runtime. These cover macOS branding/window integration, native Trash receipts and identity rechecks, upgrades, timed masks, OCR/branding guards, tracking, Range playback, scan scope, and LaMa restoration.

A fresh extraction of the 1.1.1 archive passed strict deep signature verification and the native self-test. The installed app, archive, and published source matched for all 220 packaged Python/HTML engine files.

Release ZIP SHA-256: `fb1b7c37839b28318ea2f79794559e7dcf088a72be8c23900a8b0145429a1311`.
