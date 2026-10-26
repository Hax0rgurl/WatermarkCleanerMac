# Changelog

## 2.1.0

- Add Auto, TikTok, Instagram, CapCut, and InVideo style selection. Bundle TikTok-note and Instagram-camera/Reels silhouettes; CapCut and InVideo retain named-OCR glyph profiles with adjoining text and local logo-detail checks.
- Check independently recognized watermark components on every decoded frame. Preserve visible fades and exact position changes without filling genuinely absent intervals.
- Match alternative TikTok icon arrangements so an OCR box that starts at the wordmark does not leave the music note outside the mask.
- Refresh automatic selections from older scanner versions while preserving manually corrected and removed ranges. Newly added manual areas stay additive. Save selections separately by source identity, style, and target name.
- Reject visual-player responses from a previous source or watermark selection, and revalidate the selected source after a refreshed scan before restoring frames.

## 2.0.1

- Cover the complete adjacent TikTok icon, including its colored edges and lower loop, in fresh and previously cached automatic selections.
- Preserve nonzero video starting timestamps, final-frame duration, and audio timing with exact packet-copy muxing.
- Prevent an older successful CI run from publishing a release after a newer main commit supersedes it. All build and test checks remain enabled.

## 2.0.0

- Replace DeMark/LaMa restoration with E2FGVI-HQ and ProPainter video inpainting. E2FGVI-HQ is the default.
- Automatically install a private Python 3.12.13 runtime and both model sets on first launch. Verify pinned checkpoint sizes and SHA-256 hashes, show progress, keep valid files, and retry incomplete downloads.
- Run local CPU inference through bounded overlapping video chunks and shot boundaries. Keep model predictions inside active masks before lossy encoding.
- Seed watermark tracks with locally recognized brand names. Check individual frame appearances, retain start/stop times, and keep visual player corrections available without requiring numeric timing entry.
- Preserve source files. Publish only a fully decoded and verified MP4 with matching frame count, dimensions, exact frame timestamps, and original audio packet contents.
- Remove the separate MLX/LLM setup, OpenCV production dependency, and CUDA/MMCV requirements.
- Add the shared-pipeline CLI, reproducible macOS runtime lock, real pretrained tests for both engines, native bundle validation, and Apple Silicon CI.
- Retain complete backend licenses, upstream revisions, modification notices, and checkpoint provenance. The restoration backends have noncommercial terms separate from the app's Apache license.

## Earlier releases

Versions 1.1.1 and 1.1.2 used the former DeMark/LaMa workflow. They introduced the native Mac window, named-watermark scanning, timed visual regions, completion checks, and review fixes. Their model setup, source-to-Trash behavior, and old operational instructions do not describe version 2. Git history preserves those releases.
