# Timed regions and the Python API

The desktop player lets you draw and time boxes visually. The API below is for source use or saved corrections. Both use the same local restoration and output-verification pipeline.

## Python

```python
from pathlib import Path
from watermark_cleaner.core import WatermarkCleaner

cleaner = WatermarkCleaner("e2fgvi")  # or "propainter"
cleaner.run(
    Path("input.mp4"),
    Path("cleaned.mp4"),
    watermark_regions=[
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 300, "end_frame": 360},
        {"bbox": [0.78, 0.04, 0.97, 0.13], "start_frame": 720, "end_frame": 780},
    ],
)
```

Coordinates are normalized `[left, top, right, bottom]` within `0..1`. Frame numbers are zero-based: the start is included, and the stop is excluded. These example intervals require a source with at least 780 decoded frames. A bare four-coordinate list applies throughout the video.

Pass reviewed manual regions to skip automatic scanning. Without regions, `watermark_text="InVideo, Storyblocks"` narrows the scan to those names. Use `watermark_style="tiktok"`, `"instagram"`, `"capcut"`, or `"invideo"` to choose a platform profile; the default is `"auto"`. Omitting the text uses the selected profile's platform names, or the built-in name list in Auto. Automatic scanning needs the native Apple Vision helper produced by the build script.

For example, automatic Instagram-profile processing uses:

```python
cleaner.run(
    Path("input.mp4"),
    Path("cleaned.mp4"),
    watermark_style="instagram",
)
```

The desktop player keeps saved edits separate by source identity, style, and target text. Scanner-version changes refresh automatic ranges while retaining manually changed or deleted occurrences. API callers supplying explicit regions own their saved-region provenance; those regions are validated but do not trigger a new scan.

Regions and intervals are checked against actual decoded dimensions and frame counts. Invalid coordinates, empty intervals, and out-of-range frames fail before rendering. Automatic suggestions also need the scanner's identity-check marker. That marker records how a suggestion was checked; it does not independently prove perfect detection.

## CLI JSON

Save this as `regions.json` for a source with at least 360 decoded frames:

```json
[
  {
    "bbox": [0.78, 0.04, 0.97, 0.13],
    "start_frame": 300,
    "end_frame": 360
  }
]
```

Then run:

```bash
.venv/bin/python -m watermark_cleaner.cli \
  -i input.mp4 -o cleaned.mp4 --engine propainter --regions regions.json
```

A JSON object containing a `regions` list is also accepted. An empty list is rejected, so clearing saved regions cannot silently request a new automatic scan. Region files apply to one video; batch CLI use requires automatic scanning.

## Processing and completion

The engine uses adjacent frames inside bounded chunks and avoids mixing frames across detected shot boundaries. Only active mask pixels are replaced before encoding; H.264 encoding is lossy, so an exported video is not pixel-identical outside the mask.

The writer retains decoded source frame timestamps, including variable frame rate. Output publication requires matching dimensions, frame count and timestamps, matching compressed audio contents, and a successful full video/audio decode. Originals and existing destination files are preserved. A finishing error prevents publication. Audio muxing and the final FFmpeg decode each have a 3,600-second timeout; model inference has no fixed one-hour deadline.

Output is H.264, 8-bit 4:2:0 MP4 with the source's supported audio streams copied. Odd video dimensions are currently rejected. Additional video streams, subtitles, alpha, and HDR preservation are not part of this output contract. An audio codec that cannot be copied into MP4 causes a clear finishing failure instead of silently dropping or replacing audio.
