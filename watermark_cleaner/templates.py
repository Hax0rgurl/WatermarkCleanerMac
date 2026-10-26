"""Named download-watermark profiles used by discovery, masking and the UI.

Profiles constrain independently identified branding. Selecting a platform never
turns ordinary captions into authorized watermark regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, signal

STYLE_OPTIONS = {
    "Auto": "auto",
    "TikTok": "tiktok",
    "Instagram": "instagram",
    "CapCut": "capcut",
    "InVideo": "invideo",
}

DEFAULT_NAMES = [
    "tiktok",
    "instagram",
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


@dataclass(frozen=True)
class MaskProfile:
    name: str
    label: str
    names: tuple[str, ...]
    layouts: tuple[str, ...]


PROFILES = {
    "tiktok": MaskProfile(
        "tiktok",
        "TikTok",
        ("tiktok",),
        ("note beside wordmark and creator", "note above wordmark and creator"),
    ),
    "instagram": MaskProfile(
        "instagram",
        "Instagram",
        ("instagram",),
        ("camera beside creator", "Reels icon beside creator", "Instagram wordmark and creator"),
    ),
    "capcut": MaskProfile(
        "capcut", "CapCut", ("capcut",), ("ribbon beside wordmark", "ribbon above wordmark")
    ),
    "invideo": MaskProfile(
        "invideo", "InVideo", ("invideo",), ("symbol beside wordmark", "wordmark and creator")
    ),
}


def validate_style(style: str) -> str:
    if style not in STYLE_OPTIONS.values():
        raise ValueError("Choose Auto, TikTok, Instagram, CapCut or InVideo watermark style.")
    return style


def profile_names(style: str = "auto") -> list[str]:
    style = validate_style(style)
    return DEFAULT_NAMES.copy() if style == "auto" else list(PROFILES[style].names)


@lru_cache(maxsize=128)
def _icon_template(name: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a bundled silhouette, with enough margin to compare both edges."""
    with Image.open(Path(__file__).with_name("assets") / f"{name}.png") as source:
        ink = np.asarray(source.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    ink = np.pad(ink / 255, 3)
    edges = ndimage.gaussian_filter(ink, 0.55) - ndimage.gaussian_filter(ink, 1.7)
    edges -= edges.mean()
    return edges, ink > 0.35


def icon_search_box(anchor, width: int, height: int, label: str) -> tuple[int, int, int, int]:
    """Bound companion icons to the side/above a verified wordmark or handle."""
    x1, y1, x2, y2 = anchor
    h = max(8, y2 - y1)
    above = 2.7 if label == "tiktok" else 2.2
    return (
        max(0, int(np.floor(x1 - h * 2.0))),
        max(0, int(np.floor(y1 - h * above))),
        min(width, int(np.ceil(min(x2, x1 + h * 3) + h * 0.5))),
        min(height, int(np.ceil(y2 + h * 0.6))),
    )


def match_icon(frame: np.ndarray, anchor, label: str, *, username: bool = False) -> dict | None:
    """Match a complete platform silhouette beside locally read branding.

    A handle alone is insufficient: it must have a camera/Reels silhouette.
    Matching uses shape and local contrast rather than the color white, and
    searches only a small area adjoining the independently read text.
    """
    names = {"tiktok": ("tiktok",), "instagram": ("instagram", "reels")}.get(label, ())
    if not names:
        return None
    height, width = frame.shape[:2]
    ax1, ay1, ax2, ay2 = anchor
    h = max(8, ay2 - ay1)
    left, top, right, bottom = icon_search_box(anchor, width, height, label)
    gray = frame[top:bottom, left:right].astype(np.float32).mean(axis=2)
    detail = ndimage.gaussian_filter(gray, 0.55) - ndimage.gaussian_filter(gray, 1.7)
    best = None
    sizes = range(max(8, round(h * 0.8)), max(9, round(h * 2.3)) + 1)
    for name in names:
        for size in sizes:
            template, ink = _icon_template(name, size)
            th, tw = template.shape
            if th > detail.shape[0] or tw > detail.shape[1]:
                continue
            # Zero-mean normalized correlation rejects flat white areas and
            # scene brightness changes; both light and dark variants can match.
            sums = signal.correlate(detail, np.ones_like(template), mode="valid", method="fft")
            squares = signal.correlate(
                detail * detail, np.ones_like(template), mode="valid", method="fft"
            )
            norm = np.sqrt(np.maximum(0, squares - sums * sums / template.size))
            meaningful = norm >= np.sqrt(template.size) * 1.5
            norm *= np.linalg.norm(template)
            scores = np.divide(
                signal.correlate(detail, template, mode="valid", method="fft"),
                norm,
                out=np.zeros_like(norm),
                where=meaningful,
            )
            for flat in np.argsort(np.abs(scores), axis=None)[-3:]:
                yy, xx = np.unravel_index(flat, scores.shape)
                x, y = left + int(xx) + 3, top + int(yy) + 3
                side = x + size <= ax1 + h * 1.65 and abs(y + size * 0.5 - (ay1 + ay2) * 0.5) <= h
                above = y + size <= ay1 + h * 0.35 and ax1 - h * 0.5 <= x <= min(ax2, ax1 + h * 2)
                if not (side or above):
                    continue
                score = abs(float(scores[yy, xx]))
                # Identity inference from a username requires a stronger match
                # than locating a companion to an already recognized brand.
                threshold = 0.80 if username or label == "instagram" else 0.58
                if score < threshold or (best and score <= best["score"]):
                    continue
                patch = detail[yy : yy + th, xx : xx + tw]
                polarity = 1 if scores[yy, xx] > 0 else -1
                positive = template > np.max(template) * 0.25
                if np.mean(patch[positive] * polarity > 2) < 0.60:
                    continue
                # Leave a small margin for antialiasing, drop shadows and the
                # disconnected cyan/red TikTok fringes outside the white note.
                pad = max(3, round(size * 0.20))
                best = {
                    "bbox": (
                        max(0, x - pad),
                        max(0, y - pad),
                        min(width, x + size + pad),
                        min(height, y + size + pad),
                    ),
                    "score": score,
                    "template": name,
                }
    return best
