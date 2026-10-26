"""Automatic lettering masks derived inside independently identified branding."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .timed_regions import region_boxes


def _disk(radius: int):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _part_mask(parts, width, height):
    result = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in region_boxes(parts, width, height):
        result[y1:y2, x1:x2] = True
    return result


def tiktok_icon_box(anchor, width: int, height: int) -> tuple[int, int, int, int]:
    """Enclose the complete note beside an independently recognized wordmark.

    OCR can include part of the note in its first character. The colored fringe
    and lower loop extend beyond the text baseline and may be disconnected from
    its white stem, so pixel brightness cannot establish their full extent.
    """
    x1, y1, _x2, y2 = anchor
    text_height = max(8, y2 - y1)
    return (
        max(0, int(np.floor(x1 - text_height * 0.4))),
        max(0, int(np.floor(y1 - text_height * 0.25))),
        min(width, int(np.ceil(x1 + text_height))),
        min(height, int(np.ceil(y2 + text_height * 0.35))),
    )


def build_masks(frames: list[np.ndarray], start: int, regions: list[dict]) -> np.ndarray:
    """Refine automatic boxes to glyphs; preserve manual rectangle intent."""
    height, width = frames[0].shape[:2]
    result = np.zeros((len(frames), height, width), dtype=np.uint8)
    for region in regions:
        active = [
            i
            for i in range(len(frames))
            if region["start_frame"] <= start + i < region["end_frame"]
        ]
        if not active:
            continue
        x1, y1, x2, y2 = region_boxes([region["bbox"]], width, height)[0]
        parts = region.get("mask_parts")
        if region.get("source") != "suggested" or not parts:
            for index in active:
                result[index, y1:y2, x1:x2] = 255
            continue
        full_icon = None
        if (
            str(region.get("label", "")).casefold() == "tiktok"
            and region.get("verification") == "named-ocr-v1"
        ):
            anchor = region_boxes([region.get("anchor_bbox", parts[0])], width, height)[0]
            full_icon = tiktok_icon_box(anchor, width, height)
            # Existing saved selections may contain the previous search bounds.
            # Derive the complete note from the verified anchor and expand this
            # local patch, without changing the user's stored manual rectangles.
            x1, y1 = min(x1, full_icon[0]), min(y1, full_icon[1])
            x2, y2 = max(x2, full_icon[2]), max(y2, full_icon[3])
        # At most twelve tiny context patches are stacked, even for long videos.
        samples = active[:: max(1, len(active) // 12)][:12]
        patches = np.stack([frames[i][y1:y2, x1:x2] for i in samples])
        low = np.percentile(patches, 10, axis=0).astype(np.float32)
        high = np.percentile(patches, 90, axis=0).astype(np.float32)
        median = np.median(patches, axis=0).astype(np.float32)
        gray = median.mean(axis=2)
        detail = gray - ndimage.gaussian_filter(gray, 1.7)
        stable = (high - low).max(axis=2)
        allowed = _part_mask(parts, width, height)[y1:y2, x1:x2]
        white = (low.min(axis=2) > 155) & (detail > 5) & (stable < 75)
        colored = (
            ((median.max(axis=2) - median.min(axis=2)) > 50) & (np.abs(detail) > 6) & (stable < 55)
        )
        core = (white | colored) & allowed
        # If contrast cannot reliably separate faint/dark text, keep the small
        # OCR text strips instead of masking their entire surrounding envelope.
        if np.count_nonzero(core) < max(8, np.count_nonzero(allowed) * 0.015):
            core = allowed.copy()
        icon_box = region.get("icon_search")
        if icon_box and full_icon is None:
            icon = _part_mask([icon_box], width, height)[y1:y2, x1:x2]
            candidates = (white | colored) & icon
            labels, count = ndimage.label(ndimage.binary_closing(candidates, iterations=1))
            font_height = max(10, min((p[3] - p[1]) * height for p in parts))
            for label in range(1, count + 1):
                selected = labels == label
                yy, xx = np.nonzero(selected)
                if len(xx) < 12 or len(xx) > font_height * font_height * 7:
                    continue
                if np.any(selected & allowed) or (
                    np.count_nonzero(selected & colored) >= 8
                    and xx.max() - xx.min() < font_height * 4
                    and yy.max() - yy.min() < font_height * 4
                ):
                    core |= selected
        radius = max(3, min(7, round(min((p[3] - p[1]) * height for p in parts) * 0.24)))
        glyphs = ndimage.binary_fill_holes(core)
        glyphs = ndimage.binary_dilation(glyphs, structure=_disk(radius))
        if full_icon is not None:
            left, top, right, bottom = full_icon
            glyphs[top - y1 : bottom - y1, left - x1 : right - x1] = True
        for index in active:
            result[index, y1:y2, x1:x2][glyphs] = 255
    return result


def context_crop(masks: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.any(masks > 0, axis=0))
    if not len(xs):
        raise ValueError("There is no active watermark in this video chunk.")

    def bounds(low, high, limit, minimum):
        size = min(limit, max(minimum, int(high - low + 1) + 160))
        start = max(0, min((int(low) + int(high) + 1 - size) // 2, limit - size))
        return start, start + size

    left, right = bounds(xs.min(), xs.max(), width, 320)
    top, bottom = bounds(ys.min(), ys.max(), height, 256)
    return left, top, right, bottom
