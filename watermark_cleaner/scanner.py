"""Local named-brand OCR discovery with frame-by-frame appearance checks."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import re
from typing import Callable

import numpy as np
from scipy import ndimage

from .video import iter_frames, probe_video
from .watermark_guard import LocalTextReader, matches_target, target_names

BRANDS = [
    "tiktok",
    "sora",
    "invideo",
    "kling",
    "storyblocks",
    "istock",
    "getty images",
    "veo",
    "runway",
    "pika",
    "capcut",
    "vidu",
    "luma",
]


def _box_pixels(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, int(np.floor(x1 * width))),
        max(0, int(np.floor(y1 * height))),
        min(width, int(np.ceil(x2 * width))),
        min(height, int(np.ceil(y2 * height))),
    )


def _normalized(box, width, height):
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0


def _gray(frame):
    return frame.astype(np.float32).mean(axis=2)


def _signature(patch):
    gray = _gray(patch)
    edges = gray - ndimage.gaussian_filter(gray, 1.7)
    core = (edges > 9) & (patch.min(axis=2) > 150)
    return edges, core


def _present(patch, reference):
    if patch.shape != reference.shape or not patch.size:
        return False
    expected, core = _signature(reference)
    if np.count_nonzero(core) < 8:
        # Dark branding still has a repeatable high-frequency shape.
        core = np.abs(expected) > 8
    if np.count_nonzero(core) < 8:
        return False
    actual, _ = _signature(patch)
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    correlation = float(np.sum(expected * actual)) / denominator if denominator > 1 else 0
    threshold = np.minimum(reference.min(axis=2).astype(np.float32) - 55, 185)
    bright_match = float(np.mean(patch.min(axis=2)[core] >= threshold[core]))
    return correlation >= 0.40 and bright_match >= 0.50


def _candidate(anchor, texts, width, height, label):
    box = _box_pixels(anchor["bbox"], width, height)
    x1, y1, x2, y2 = box
    h = max(8, y2 - y1)
    parts = [box]
    for text in texts:
        if text is anchor or text.get("confidence", 0) < 0.5:
            continue
        value = str(text.get("text", "")).strip()
        other = _box_pixels(text["bbox"], width, height)
        ox1, oy1, ox2, oy2 = other
        adjacent = oy1 >= y1 - h * 0.3 and oy1 <= y2 + h * 2 and ox1 < x2 + h and ox2 > x1 - h
        if adjacent and (value.startswith("@") or re.match(r"^[@©]\s*\w", value)):
            parts.append(other)
    # This search area is used only for persistent logo pixels, never a solid box.
    # TikTok's music note sits against the wordmark. Searching several text
    # heights above it can mistake bright scene details for part of the logo.
    if label == "tiktok":
        icon = (
            max(0, x1 - h * 0.6),
            max(0, y1 - h * 0.6),
            min(width, x1 + h * 1.4),
            min(height, y2 + h * 0.2),
        )
    else:
        icon = (
            max(0, x1 - h),
            max(0, y1 - h * 1.5),
            min(width, x1 + h * 2),
            min(height, y2 + h * 0.2),
        )
    combined = (
        min(p[0] for p in parts),
        min(p[1] for p in parts),
        max(p[2] for p in parts),
        max(p[3] for p in parts),
    )
    envelope = (
        min(combined[0], icon[0]),
        min(combined[1], icon[1]),
        max(combined[2], icon[2]),
        max(combined[3], icon[3]),
    )
    envelope = (
        max(0, int(envelope[0]) - 6),
        max(0, int(envelope[1]) - 6),
        min(width, int(envelope[2]) + 6),
        min(height, int(envelope[3]) + 6),
    )
    return {
        "bbox": envelope,
        "anchor": box,
        "parts": parts,
        "icon": tuple(map(int, icon)),
        "label": label,
    }


def scan_video(
    path: Path,
    watermark_text: str = "",
    progress: Callable[[float, str], None] | None = None,
    *,
    reader=None,
) -> dict:
    """Find the requested branding without treating arbitrary white captions as marks.

    Common names are scanned automatically. Custom names remain explicit. OCR
    seeds identities; a local pixel-template test checks every decoded frame.
    """
    info = probe_video(path)
    names = target_names(watermark_text) if watermark_text.strip() else BRANDS
    sample = set(info.scene_boundaries[:-1]) | {end - 1 for end in info.scene_boundaries[1:]}
    moment = info.frame_times[0]
    while moment <= info.frame_times[-1]:
        sample.add(min(info.total_frames - 1, bisect_left(info.frame_times, moment)))
        moment += 0.25
    tracks = []
    own_reader = reader is None
    reader = reader or LocalTextReader()
    try:
        for index, frame in enumerate(iter_frames(path)):
            if index not in sample:
                continue
            texts = reader.read(frame)
            for text in texts:
                value = str(text.get("text", ""))
                if text.get("confidence", 0) < 0.5 or not matches_target(value, names):
                    continue
                bbox = text.get("bbox", [])
                if len(bbox) != 4:
                    continue
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.04 or len(value) > 60:
                    continue
                # Full captions mentioning a brand are not platform branding.
                if not watermark_text.strip() and len(re.findall(r"\w+", value)) > 3:
                    continue
                label = next((name for name in names if matches_target(value, [name])), value)
                candidate = _candidate(text, texts, info.width, info.height, label)
                track = next(
                    (
                        t
                        for t in tracks
                        if t["label"] == label and _iou(t["anchor"], candidate["anchor"]) > 0.45
                    ),
                    None,
                )
                if track is None:
                    x1, y1, x2, y2 = candidate["anchor"]
                    tracks.append(
                        {**candidate, "reference": frame[y1:y2, x1:x2].copy(), "observed": {index}}
                    )
                else:
                    track["observed"].add(index)
                    # Creator lines occasionally read more completely in another frame.
                    for part in candidate["parts"]:
                        if all(_iou(part, p) < 0.5 for p in track["parts"]):
                            track["parts"].append(part)
                    track["bbox"] = (
                        min(track["bbox"][0], candidate["bbox"][0]),
                        min(track["bbox"][1], candidate["bbox"][1]),
                        max(track["bbox"][2], candidate["bbox"][2]),
                        max(track["bbox"][3], candidate["bbox"][3]),
                    )
            if progress:
                progress(
                    0.48 * (index + 1) / info.total_frames, "Finding watermark branding locally"
                )
    finally:
        if own_reader:
            reader.close()
    presence = [np.zeros(info.total_frames, dtype=bool) for _ in tracks]
    for index, frame in enumerate(iter_frames(path)):
        for track, active in zip(tracks, presence, strict=True):
            x1, y1, x2, y2 = track["anchor"]
            active[index] = index in track["observed"] or _present(
                frame[y1:y2, x1:x2], track["reference"]
            )
        if progress and index % 12 == 0:
            progress(
                0.48 + 0.52 * (index + 1) / info.total_frames,
                "Checking every watermark start and stop frame",
            )
    regions = []
    for number, (track, active) in enumerate(zip(tracks, presence, strict=True)):
        padded = np.pad(active.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for start, end in zip(starts, ends, strict=True):
            regions.append(
                {
                    "id": f"auto-{number}-{int(start)}",
                    "bbox": _normalized(track["bbox"], info.width, info.height),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "label": track["label"],
                    "source": "suggested",
                    "verification": "named-ocr-v1",
                    "timing_verified": True,
                    "mask_parts": [_normalized(p, info.width, info.height) for p in track["parts"]],
                    "anchor_bbox": _normalized(track["anchor"], info.width, info.height),
                    "icon_search": _normalized(track["icon"], info.width, info.height),
                }
            )
    regions.sort(key=lambda r: (r["start_frame"], r["bbox"][0]))
    if progress:
        progress(1, "Watermark scan checked")
    return {
        "regions": regions,
        "total_frames": info.total_frames,
        "fps": info.fps,
        "duration": info.duration,
        "summary": f"Found {len(regions)} checked watermark appearance(s).",
        "timing_verified": True,
        "model": "Local text recognition and temporal pixel verification",
        "semantic_model_check": False,
    }
