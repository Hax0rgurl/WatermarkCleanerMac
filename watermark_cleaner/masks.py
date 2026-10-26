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
        if icon_box:
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
