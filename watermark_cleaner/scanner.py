"""Local named-brand OCR discovery with frame-by-frame appearance checks."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import re
from typing import Callable

import numpy as np
from scipy import ndimage

from .masks import tiktok_icon_box
from .templates import DEFAULT_NAMES, match_icon, profile_names
from .video import iter_frames, probe_video
from .watermark_guard import LocalTextReader, matches_target, target_names

BRANDS = DEFAULT_NAMES.copy()
SCAN_VERSION = "platform-templates-v1"
MAX_COMPONENT_REFERENCES = 8


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


def _reference(reference):
    expected, core = _signature(reference)
    if np.count_nonzero(core) < 8:
        core = np.abs(expected) > 8
    support = ndimage.binary_dilation(core, iterations=1)
    return {
        "shape": reference.shape,
        "edges": expected,
        "core": core,
        "support": support,
        "norm": float(np.linalg.norm(expected)),
        "support_norm": float(np.linalg.norm(expected[support])),
        "brightness": np.minimum(reference.min(axis=2).astype(np.float32) - 55, 185),
    }


def _matches_reference(actual, minimum, reference):
    core = reference["core"]
    if np.count_nonzero(core) < 8:
        return False
    expected = reference["edges"]
    denominator = reference["norm"] * float(np.linalg.norm(actual))
    correlation = float(np.sum(expected * actual)) / denominator if denominator > 1 else 0
    bright_match = float(np.mean(minimum[core] >= reference["brightness"][core]))
    if correlation >= 0.40 and bright_match >= 0.50:
        return True
    # Background motion can overwhelm a whole text rectangle. Compare only the
    # verified glyphs and their immediate edge surroundings. Relative contrast
    # retains a fading component while its shape is still present, without
    # filling genuinely absent intervals or accepting arbitrary bright text.
    support = reference["support"]
    cropped = actual[support]
    actual_norm = float(np.linalg.norm(cropped))
    denominator = reference["support_norm"] * actual_norm
    shape_match = float(np.sum(expected[support] * cropped)) / denominator if denominator > 1 else 0
    if shape_match < 0.70 or actual_norm < reference["support_norm"] * 0.12:
        return False
    if np.any(expected[core] < 0):
        agreement = float(np.mean(actual[core] * expected[core] > np.square(expected[core]) * 0.15))
    else:
        agreement = float(np.mean(actual[core] > np.maximum(2, expected[core] * 0.15)))
    return agreement >= 0.55


def _present(patch, reference):
    if patch.shape != reference.shape or not patch.size:
        return False
    return _matches_reference(_signature(patch)[0], patch.min(axis=2), _reference(reference))


def _remember_components(track, candidate, frame):
    """Learn OCR lettering and independently matched platform silhouettes."""
    for box in candidate["parts"]:
        component = next((c for c in track["components"] if _iou(c["box"], box) > 0.45), None)
        if component is None:
            component = {"box": box, "references": []}
            track["components"].append(component)
        x1, y1, x2, y2 = component["box"]
        patch = frame[y1:y2, x1:x2]
        reference = _reference(patch)
        if any(
            old["norm"] * reference["norm"] > 1
            and float(np.sum(old["edges"] * reference["edges"])) / (old["norm"] * reference["norm"])
            > 0.94
            for old in component["references"]
        ):
            continue
        if len(component["references"]) < MAX_COMPONENT_REFERENCES:
            component["references"].append(reference)


def _components_present(frame, track):
    for component in track["components"]:
        x1, y1, x2, y2 = component["box"]
        patch = frame[y1:y2, x1:x2]
        actual = _signature(patch)[0]
        minimum = patch.min(axis=2)
        if any(_matches_reference(actual, minimum, r) for r in component["references"]):
            return True
    return False


def _adjoining(first, second):
    h = max(first[3] - first[1], second[3] - second[1], 8)
    horizontal_gap = max(0, first[0] - second[2], second[0] - first[2])
    vertical_gap = max(0, first[1] - second[3], second[1] - first[3])
    return horizontal_gap <= h and vertical_gap <= h * 2.5


def _creator_handle(value):
    # OCR can read a camera/Reels stroke as a few punctuation marks before @.
    # Words, suffix captions and long punctuation runs are never creator-only marks.
    match = re.fullmatch(r"[^\w@]{0,6}(@\w[\w.]{1,59})", str(value).strip())
    return match.group(1) if match else None


def _has_named_companion(box, texts, tracks, frame, width, height):
    """Do not relabel a known platform's creator as an Instagram-only mark."""
    for text in texts:
        other = text.get("bbox", [])
        if (
            text.get("confidence", 0) >= 0.5
            and len(other) == 4
            and matches_target(str(text.get("text", "")), BRANDS)
            and (other[2] - other[0]) * (other[3] - other[1]) <= 0.04
            and len(re.findall(r"\w+", str(text.get("text", "")))) <= 3
            and _adjoining(box, _box_pixels(other, width, height))
        ):
            return True
    # OCR can miss the wordmark in a later sample while reading its creator.
    # The already verified live wordmark still identifies the platform.
    return any(
        track["label"] != "instagram"
        and _adjoining(box, track["anchor"])
        and _components_present(frame, track)
        for track in tracks
    )


def _platform_companion(anchor, texts, label, width, height):
    """Use a nearby independently read wordmark for a selected creator profile."""
    box = _box_pixels(anchor["bbox"], width, height)
    candidates = [
        text
        for text in texts
        if text.get("confidence", 0) >= 0.5
        and len(text.get("bbox", [])) == 4
        and matches_target(str(text.get("text", "")), [label])
        and len(re.findall(r"\w+", str(text.get("text", "")))) <= 3
        and (text["bbox"][2] - text["bbox"][0]) * (text["bbox"][3] - text["bbox"][1]) <= 0.04
        and _adjoining(box, _box_pixels(text["bbox"], width, height))
    ]
    return min(
        candidates,
        key=lambda text: sum(
            abs(a - b) for a, b in zip(box, _box_pixels(text["bbox"], width, height), strict=True)
        ),
        default=None,
    )


def _merge_creator_profile_tracks(tracks, names, style):
    """Unify a creator and its later-identified platform at the same text block.

    Early OCR may read the creator while misreading the platform lettering.
    Shared OCR text geometry proves the connection; proximity alone does not.
    """
    eligible = {*names, style}

    def same_text(first, second):
        return (
            first["label"] in eligible
            and second["label"] in eligible
            and any(
                _iou(a, b) > 0.5
                for a in first["parts"]
                if not first["icon_verified"] or a != first["icon"]
                for b in second["parts"]
                if not second["icon_verified"] or b != second["icon"]
            )
        )

    remaining, result = list(tracks), []
    while remaining:
        group = [remaining.pop(0)]
        index = 0
        while index < len(remaining):
            if any(same_text(known, remaining[index]) for known in group):
                group.append(remaining.pop(index))
                index = 0
            else:
                index += 1
        if len(group) == 1:
            result.append(group[0])
            continue
        preferred = min(
            group, key=lambda track: (track["label"] != style, track["anchor_is_creator"])
        )
        merged = {**preferred, "parts": [], "components": [], "observed": set()}
        for track in group:
            merged["components"].extend(track["components"])
            merged["observed"].update(track["observed"])
            for part in track["parts"]:
                if track["icon_verified"] and part == track["icon"]:
                    continue
                if part not in merged["parts"]:
                    merged["parts"].append(part)
        boxes = [track["bbox"] for track in group]
        merged["bbox"] = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        icons = [track["icon"] for track in group if track["icon_verified"]]
        if icons:
            merged["icon"] = (
                min(box[0] for box in icons),
                min(box[1] for box in icons),
                max(box[2] for box in icons),
                max(box[3] for box in icons),
            )
            merged["icon_verified"] = True
            merged["parts"].append(merged["icon"])
        result.append(merged)
    return result


def _candidate(anchor, texts, width, height, label, frame=None):
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
        if adjacent and (_creator_handle(value) or re.match(r"^[@©]\s*\w", value)):
            parts.append(other)
    # The named TikTok identity authorizes its complete adjacent note, including
    # disconnected colored fringes below the text baseline. Other brands retain
    # a search area for persistent glyph components.
    if label == "tiktok":
        icon = tiktok_icon_box(box, width, height)
    else:
        icon = (
            max(0, x1 - h),
            max(0, y1 - h * 1.5),
            min(width, x1 + h * 2),
            min(height, y2 + h * 0.2),
        )
    matched = (
        match_icon(frame, box, label, username=_creator_handle(anchor.get("text", "")) is not None)
        if frame is not None
        else None
    )
    if matched is not None:
        ix1, iy1, ix2, iy2 = matched["bbox"]
        icon = (max(0, ix1 - 2), max(0, iy1 - 2), min(width, ix2 + 2), min(height, iy2 + 2))
        parts.append(icon)
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
        "icon_verified": matched is not None,
        "anchor_is_creator": _creator_handle(anchor.get("text", "")) is not None,
        "label": label,
    }


def scan_video(
    path: Path,
    watermark_text: str = "",
    progress: Callable[[float, str], None] | None = None,
    *,
    reader=None,
    watermark_style: str = "auto",
) -> dict:
    """Find the requested branding without treating arbitrary white captions as marks.

    Common names are scanned automatically. Custom names remain explicit. OCR
    seeds identities; a local pixel-template test checks every decoded frame.
    """
    info = probe_video(path)
    defaults = profile_names(watermark_style)
    names = target_names(watermark_text) if watermark_text.strip() else defaults
    allow_instagram_creators = watermark_style in {"auto", "instagram"} and "instagram" in names
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
                if text.get("confidence", 0) < 0.5:
                    continue
                bbox = text.get("bbox", [])
                if len(bbox) != 4:
                    continue
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.04 or len(value) > 60:
                    continue
                # Full captions mentioning a brand are not platform branding.
                if not watermark_text.strip() and len(re.findall(r"\w+", value)) > 3:
                    continue
                label = next((name for name in names if matches_target(value, [name])), None)
                anchor = text
                if label is not None and label not in BRANDS and watermark_style != "auto":
                    companion = _platform_companion(
                        text, texts, watermark_style, info.width, info.height
                    )
                    tracked = next(
                        (
                            track
                            for track in tracks
                            if track["label"] == watermark_style
                            and _adjoining(
                                _box_pixels(bbox, info.width, info.height), track["anchor"]
                            )
                            and _components_present(frame, track)
                        ),
                        None,
                    )
                    if companion is not None:
                        anchor, label = companion, watermark_style
                    elif tracked is not None:
                        anchor = {
                            "text": watermark_style,
                            "bbox": _normalized(tracked["anchor"], info.width, info.height),
                            "confidence": 1,
                        }
                        label = watermark_style
                    elif (
                        match_icon(
                            frame,
                            _box_pixels(bbox, info.width, info.height),
                            watermark_style,
                            username=_creator_handle(value) is not None,
                        )
                        is not None
                    ):
                        # A handle without a wordmark needs the stricter
                        # independent silhouette match before adopting a brand.
                        label = watermark_style
                if label is None and allow_instagram_creators:
                    # Instagram downloads can show only a camera/Reels symbol
                    # beside a creator. A handle alone never authorizes a mask.
                    edge = bbox[0] <= 0.18 or bbox[2] >= 0.82 or bbox[1] <= 0.12 or bbox[3] >= 0.88
                    if (
                        edge
                        and _creator_handle(value) is not None
                        and not _has_named_companion(
                            _box_pixels(bbox, info.width, info.height),
                            texts,
                            tracks,
                            frame,
                            info.width,
                            info.height,
                        )
                        and match_icon(
                            frame,
                            _box_pixels(bbox, info.width, info.height),
                            "instagram",
                            username=True,
                        )
                        is not None
                    ):
                        label = "instagram"
                if label is None:
                    continue
                candidate = _candidate(anchor, texts, info.width, info.height, label, frame=frame)
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
                    track = {**candidate, "components": [], "observed": {index}}
                    tracks.append(track)
                else:
                    track["observed"].add(index)
                    # Creator lines occasionally read more completely in another frame.
                    for part in candidate["parts"]:
                        if candidate["icon_verified"] and part == candidate["icon"]:
                            continue
                        if all(_iou(part, p) < 0.5 for p in track["parts"]):
                            track["parts"].append(part)
                    if candidate["icon_verified"]:
                        icon = candidate["icon"]
                        if track["icon_verified"]:
                            previous = track["icon"]
                            track["parts"] = [p for p in track["parts"] if p != previous]
                            icon = (
                                min(previous[0], icon[0]),
                                min(previous[1], icon[1]),
                                max(previous[2], icon[2]),
                                max(previous[3], icon[3]),
                            )
                        track["icon"] = icon
                        track["icon_verified"] = True
                        track["parts"].append(icon)
                    track["bbox"] = (
                        min(track["bbox"][0], candidate["bbox"][0]),
                        min(track["bbox"][1], candidate["bbox"][1]),
                        max(track["bbox"][2], candidate["bbox"][2]),
                        max(track["bbox"][3], candidate["bbox"][3]),
                    )
                _remember_components(track, candidate, frame)
            if progress:
                progress(
                    0.48 * (index + 1) / info.total_frames, "Finding watermark branding locally"
                )
    finally:
        if own_reader:
            reader.close()
    if watermark_text.strip() and watermark_style != "auto":
        tracks = _merge_creator_profile_tracks(tracks, names, watermark_style)
    presence = [np.zeros(info.total_frames, dtype=bool) for _ in tracks]
    for index, frame in enumerate(iter_frames(path)):
        for track, active in zip(tracks, presence, strict=True):
            active[index] = index in track["observed"] or _components_present(frame, track)
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
                    "timing_method": "verified-component-templates-v2",
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
        "scan_version": SCAN_VERSION,
        "watermark_style": watermark_style,
    }
