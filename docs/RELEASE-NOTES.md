# MAC WATERMARK CLEANER 2.1.0

Version 2.1.0 fixes gaps at watermark transitions and incomplete TikTok logo masks. It adds optional platform styles, local Instagram camera/Reels matching beside creator text, and automatic refresh of old scan results while retaining manual corrections.

Version 2 replaces the former DeMark/LaMa restoration pipeline with two local video models: E2FGVI-HQ, selected by default, and ProPainter.

Opening the app automatically installs its private runtime and downloads both verified model sets. Later launches reuse valid files. There are no paid processing credits or video uploads.

Choose videos, choose an engine, and remove the detected watermarks. Local named-brand scanning checks appearance times; the optional source player lets you scrub, draw corrections, and set start/stop points at the playhead. Detailed reconstruction still depends on the source and mask.

The app keeps originals and creates new MP4 files. Before reporting completion, it checks frame counts, dimensions, exact decoded frame timestamps, original audio packet contents, and full video/audio decoding. Both pretrained engines and the native app build are exercised by CI before this release job can run.

Requires Apple Silicon and macOS 14 or later. Unzip the app into Applications. Initial setup needs internet access for Python, dependencies, and approximately 364 MB of model checkpoints. The app is ad-hoc signed, not Apple-notarized.

The original app code is Apache-2.0; the bundled E2FGVI and ProPainter backends have separate noncommercial licenses. See the complete third-party notices in the source and app bundle.

The release ZIP and SHA-256 file come from the successful macOS job for the tagged commit. Published releases are not replaced by reruns.
