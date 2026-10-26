"""Platform silhouettes must identify complete logos without adopting captions."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from watermark_cleaner.masks import build_masks
from watermark_cleaner.scanner import _candidate, _normalized
from watermark_cleaner.templates import match_icon


ASSETS = Path(__file__).resolve().parents[1] / "watermark_cleaner/assets"


def _stamp(image, name, point, size, color):
    with Image.open(ASSETS / f"{name}.png") as source:
        alpha = source.resize((size, size), Image.Resampling.LANCZOS)
    x, y = point
    image.paste((color,) * 3, (x, y, x + size, y + size), alpha)
    truth = np.zeros((image.height, image.width), bool)
    truth[y : y + size, x : x + size] = np.asarray(alpha) > 8
    return truth


@pytest.mark.parametrize("name", ["instagram", "reels"])
@pytest.mark.parametrize("background,foreground", [(35, 255), (215, 0)])
def test_complete_camera_and_reels_on_flat_light_and_dark_backgrounds(name, background, foreground):
    image = Image.new("RGB", (240, 140), (background,) * 3)
    truth = _stamp(image, name, (20, 60), 24, foreground)
    found = match_icon(np.asarray(image), (54, 64, 210, 83), "instagram", username=True)
    assert found and found["template"] == name and found["score"] > 0.95
    x1, y1, x2, y2 = found["bbox"]
    selected = np.zeros_like(truth)
    selected[y1:y2, x1:x2] = True
    assert np.all(selected[truth])
    assert selected.sum() < 40 * 40


@pytest.mark.parametrize("name", ["instagram", "tiktok"])
def test_flat_and_caption_edges_are_not_complete_logos(name):
    frame = np.full((160, 240, 3), 35, np.uint8)
    assert match_icon(frame, (80, 94, 140, 103), name) is None
    frame[94:103, 80:140] = 255
    assert match_icon(frame, (80, 94, 140, 103), name) is None


@pytest.mark.parametrize("layout", ["side", "above"])
@pytest.mark.parametrize("ocr_includes_icon", [False, True])
def test_tiktok_masks_whole_note_for_both_layouts_and_ocr_box_variants(layout, ocr_includes_icon):
    image = Image.new("RGB", (320, 240), (38, 38, 38))
    point = (55, 147) if layout == "side" else (95, 111)
    truth = _stamp(image, "tiktok", point, 31, 245)
    ImageDraw.Draw(image).text(
        (95, 151), "TikTok", font=ImageFont.load_default(size=22), fill="white"
    )
    anchor = (point[0] if ocr_includes_icon and layout == "side" else 95, 151, 170, 175)
    text = {"text": "TikTok", "bbox": _normalized(anchor, 320, 240), "confidence": 0.99}
    frame = np.asarray(image)
    candidate = _candidate(text, [text], 320, 240, "tiktok", frame=frame)
    region = {
        "bbox": _normalized(candidate["bbox"], 320, 240),
        "anchor_bbox": text["bbox"],
        "icon_search": _normalized(candidate["icon"], 320, 240),
        "mask_parts": [_normalized(p, 320, 240) for p in candidate["parts"]],
        "label": "tiktok",
        "source": "suggested",
        "verification": "named-ocr-v1",
        "start_frame": 1,
        "end_frame": 3,
    }
    masks = build_masks([frame] * 4, 0, [region])
    assert np.all(masks[1:3, truth] == 255)
    assert not masks[[0, 3]].any()
    assert not masks[:, :70].any()


def test_named_instagram_does_not_turn_generic_logo_search_into_solid_mask():
    frame = np.full((160, 240, 3), 35, np.uint8)
    frame[100:110:2, 100:180] = 245
    region = {
        "bbox": [0.2, 0.35, 0.8, 0.75],
        "mask_parts": [[100 / 240, 100 / 160, 180 / 240, 110 / 160]],
        "anchor_bbox": [100 / 240, 100 / 160, 180 / 240, 110 / 160],
        "icon_search": [80 / 240, 75 / 160, 120 / 240, 112 / 160],
        "label": "instagram",
        "source": "suggested",
        "verification": "named-ocr-v1",
        "start_frame": 0,
        "end_frame": 1,
    }
    mask = build_masks([frame], 0, [region])[0]
    assert mask[100:110, 100:180].any()
    assert not mask[:85].any()
