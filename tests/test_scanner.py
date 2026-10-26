"""Automatic discovery identifies named branding and checks actual frame ranges."""

from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest

from watermark_cleaner import scanner
from watermark_cleaner.masks import build_masks, context_crop
from watermark_cleaner.watermark_guard import LocalTextReader, matches_target


def _frames(count=16):
    frames = [np.full((160, 240, 3), 45, np.uint8) for _ in range(count)]
    # Distinct high-frequency shapes stand in for lettering without using OCR
    # output to decide whether lettering is present in unsampled frames.
    for index, frame in enumerate(frames):
        frame[0, 0, 0] = index
        if index < 5 or 9 <= index < 13:
            frame[50:60:2, 20:60] = 245
            frame[51:60:2, 20:60:3] = 245
        if index >= 5:
            frame[105:115:2, 170:210] = 245
            frame[106:115:2, 170:210:3] = 245
    return frames


class Reader:
    def __init__(self, texts):
        self.texts = texts
        self.seen = []

    def read(self, frame):
        index = int(frame[0, 0, 0])
        self.seen.append(index)
        return self.texts(index)


def _text(value, box, confidence=0.99):
    return {"text": value, "bbox": box, "confidence": confidence}


@pytest.fixture
def video(monkeypatch):
    frames = _frames()
    info = SimpleNamespace(
        width=240,
        height=160,
        total_frames=len(frames),
        fps=8,
        duration=2,
        frame_times=[i / 8 for i in range(len(frames))],
        scene_boundaries=[0, len(frames)],
    )
    monkeypatch.setattr(scanner, "probe_video", lambda path: info)
    monkeypatch.setattr(scanner, "iter_frames", lambda path: iter(frames))
    return frames


def test_names_seed_exact_start_stop_and_simultaneous_marks(video):
    def texts(index):
        result = []
        if index < 5 or 9 <= index < 13:
            result.append(_text("TikTok", [20 / 240, 50 / 160, 60 / 240, 60 / 160]))
        if index >= 5:
            result.append(_text("Kling", [170 / 240, 105 / 160, 210 / 240, 115 / 160]))
        return result

    reader = Reader(texts)
    result = scanner.scan_video(Path("unused"), reader=reader)
    ranges = [(r["label"], r["start_frame"], r["end_frame"]) for r in result["regions"]]
    assert ranges == [("tiktok", 0, 5), ("kling", 5, 16), ("tiktok", 9, 13)]
    assert 5 not in reader.seen  # Boundaries are found between OCR samples.
    assert all(r["timing_verified"] and r["mask_parts"] for r in result["regions"])


def test_white_captions_and_brand_mentions_do_not_become_masks(video):
    reader = Reader(
        lambda _: [
            _text("LOOK AT THESE WHITE WORDS", [0.1, 0.2, 0.6, 0.23]),
            _text("I made this on TikTok yesterday", [0.1, 0.4, 0.9, 0.43]),
            _text("Soraya", [0.1, 0.5, 0.3, 0.53]),
            _text("TikTok", [0.0, 0.0, 0.9, 0.9]),
            _text("Kling", [0.1, 0.7, 0.3, 0.73], 0.1),
        ]
    )
    assert scanner.scan_video(Path("unused"), reader=reader)["regions"] == []


def test_custom_name_is_explicit_and_reader_failure_is_not_no_matches(video):
    reader = Reader(lambda _: [_text("ExampleBrand", [20 / 240, 50 / 160, 60 / 240, 60 / 160])])
    assert not scanner.scan_video(Path("unused"), reader=reader)["regions"]
    assert scanner.scan_video(Path("unused"), "ExampleBrand", reader=reader)["regions"]

    class Broken:
        def read(self, frame):
            raise RuntimeError("local recognition failed")

    with pytest.raises(RuntimeError, match="recognition failed"):
        scanner.scan_video(Path("unused"), reader=Broken())


def test_tiktok_icon_search_stays_against_the_confirmed_wordmark():
    anchor = _text("TikTok", [0.4, 0.5, 0.6, 0.55])
    result = scanner._candidate(anchor, [anchor], 1000, 1000, "tiktok")
    assert result["icon"][1] >= 470
    assert result["icon"][0] >= 370


def test_manual_masks_are_exact_and_end_exclusive():
    frames = [np.zeros((160, 240, 3), np.uint8) for _ in range(5)]
    region = {"bbox": [0.1, 0.2, 0.3, 0.4], "start_frame": 11, "end_frame": 13, "source": "manual"}
    masks = build_masks(frames, 10, [region])
    assert not masks[[0, 3, 4]].any()
    assert np.all(masks[1:3, 32:64, 24:72] == 255)
    assert np.count_nonzero(masks) == 2 * 32 * 48
    assert context_crop(masks, 240, 160) == (0, 0, 240, 160)


def test_automatic_glyph_mask_excludes_unrelated_scene_details():
    frames = [np.zeros((160, 240, 3), np.uint8) for _ in range(12)]
    for frame in frames:
        frame[100:110:2, 120:160] = 250
        frame[65:73, 110:118] = [0, 20, 250]  # Bright scene detail above the logo.
    anchor = _text("TikTok", [0.5, 0.625, 160 / 240, 110 / 160])
    candidate = scanner._candidate(anchor, [anchor], 240, 160, "tiktok")
    region = {
        "bbox": scanner._normalized(candidate["bbox"], 240, 160),
        "mask_parts": [anchor["bbox"]],
        "icon_search": scanner._normalized(candidate["icon"], 240, 160),
        "start_frame": 0,
        "end_frame": 12,
        "source": "suggested",
    }
    masks = build_masks(frames, 0, [region])
    assert masks[:, 100:110, 120:160].any()
    assert not masks[:, :80].any()
    assert masks[0].sum() < 255 * 1800


def test_real_native_reader_recognizes_branding_and_caption_separately():
    from PIL import Image, ImageDraw, ImageFont

    helper = Path(
        os.environ.get(
            "MAC_WATERMARK_OCR_HELPER",
            str(Path(__file__).resolve().parents[1] / ".build/macos/vision-ocr"),
        )
    )
    assert helper.is_file(), "Build the native app before running the integration suite."
    image = Image.new("RGB", (1024, 576), (35, 45, 60))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    draw.text((40, 60), "FOCUS ON THE STORY", fill="white", font=font)
    draw.text((750, 480), "TikTok", fill="white", font=font)
    reader = LocalTextReader(helper)
    try:
        texts = reader.read(np.asarray(image)[:, :, ::-1])
    finally:
        reader.close()
    assert any(matches_target(t["text"], ["tiktok"]) for t in texts)
    assert any("FOCUS" in t["text"] for t in texts)
    assert not any(matches_target(t["text"], scanner.BRANDS) for t in texts if "FOCUS" in t["text"])
