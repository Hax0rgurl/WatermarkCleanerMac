# Changelog

## 1.1.1

- Added automatic named-brand scanning, OCR checks, model-confirmed visual tracking, and audited start/stop ranges.
- Added playable video, scrubbing, frame stepping, occurrence navigation, interval playback, and playhead-based manual corrections.
- Restricted automatic masks to matching named branding to reduce accidental caption removal.
- Fixed zero-result scans caused by copied icon descriptions and brand/account strings.
- Added direct MLX fallback when a configured local vision server is unavailable.
- Preserved checked timelines when model settings change; explicit rescans replace suggestions while preserving manual edits.
- Improved LaMa mask compositing and local restoration context, preserving pixels outside selected masks before encoding.
- Added identity-pinned HTTP Range playback and completion verification of all video/audio packets.
- Added native Apple Vision OCR tooling and kept the optional vision environment separate from restoration dependencies.

See [verification](docs/VERIFICATION.md) for tested results and remaining limitations.
