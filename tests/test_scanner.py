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
    region = {
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "start_frame": 11,
        "end_frame": 13,
        "source": "manual",
        "label": "tiktok",
        "verification": "named-ocr-v1",
        "anchor_bbox": [0.1, 0.2, 0.3, 0.4],
        "mask_parts": [[0.1, 0.2, 0.3, 0.4]],
    }
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


def test_named_tiktok_masks_complete_disconnected_note_and_lower_colored_fringe():
    frames = [np.zeros((160, 240, 3), np.uint8) for _ in range(4)]
    note = np.zeros((160, 240), dtype=bool)
    note[78:96, 110:114] = True  # White stem, separate from the loop.
    note[96:102, 98:105] = True  # White lower loop.
    note[99:106, 94:97] = True  # Cyan fringe below the text baseline.
    note[103:106, 116:119] = True  # Disconnected red fringe.
    for frame in frames:
        frame[78:96, 110:114] = 245
        frame[96:102, 98:105] = 245
        frame[99:106, 94:97] = [245, 245, 10]
        frame[103:106, 116:119] = [10, 20, 245]
        frame[50:56, 100:106] = [0, 30, 250]  # Unrelated scene detail above.
        frame[82:98:2, 126:177] = 245
    region = {
        "bbox": [92 / 240, 72 / 160, 185 / 240, 110 / 160],
        "mask_parts": [[100 / 240, 80 / 160, 180 / 240, 100 / 160]],
        "anchor_bbox": [100 / 240, 80 / 160, 180 / 240, 100 / 160],
        "icon_search": [95 / 240, 75 / 160, 120 / 240, 101 / 160],
        "start_frame": 11,
        "end_frame": 13,
        "label": "tiktok",
        "source": "suggested",
        "verification": "named-ocr-v1",
    }
    masks = build_masks(frames, 10, [region])
    assert np.all(masks[1:3, note] == 255)
    assert not masks[[0, 3]].any()
    assert not masks[:, :65].any()


def test_cached_tiktok_geometry_expands_its_patch_from_the_verified_anchor():
    frames = [np.zeros((160, 240, 3), np.uint8)]
    # Schema3 selections from2.0.0 may have a narrow/obsolete icon search and
    # envelope. The stored independently verified anchor remains authoritative.
    region = {
        "bbox": [100 / 240, 80 / 160, 180 / 240, 100 / 160],
        "mask_parts": [[100 / 240, 80 / 160, 180 / 240, 100 / 160]],
        "anchor_bbox": [100 / 240, 80 / 160, 180 / 240, 100 / 160],
        "icon_search": [108 / 240, 82 / 160, 110 / 240, 84 / 160],
        "start_frame": 0,
        "end_frame": 1,
        "label": "TikTok",
        "source": "suggested",
        "verification": "named-ocr-v1",
    }
    masks = build_masks(frames, 0, [region])
    assert np.all(masks[0, 75:107, 92:120] == 255)
    assert not masks[0, :70].any()
    assert not masks[0, 110:].any()
    assert region["bbox"] == [100 / 240, 80 / 160, 180 / 240, 100 / 160]


def test_full_icon_envelope_requires_named_tiktok_identity():
    frame = np.zeros((160, 240, 3), np.uint8)
    frame[82:98:2, 126:177] = 245
    region = {
        "bbox": [90 / 240, 70 / 160, 185 / 240, 110 / 160],
        "mask_parts": [[100 / 240, 80 / 160, 180 / 240, 100 / 160]],
        "anchor_bbox": [100 / 240, 80 / 160, 180 / 240, 100 / 160],
        "icon_search": [92 / 240, 75 / 160, 120 / 240, 107 / 160],
        "start_frame": 0,
        "end_frame": 1,
        "label": "custom brand",
        "source": "suggested",
        "verification": "named-ocr-v1",
    }
    masks = build_masks([frame], 0, [region])
    assert not masks[0, 103:106, 94:97].any()
    unverified = {**region, "label": "tiktok", "verification": "old"}
    assert not build_masks([frame], 0, [unverified])[0, 103:106, 94:97].any()


def test_glyph_shape_survives_a_moving_background_inside_its_ocr_rectangle():
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage

    lettering = Image.new("L", (180, 32))
    ImageDraw.Draw(lettering).text((5, 3), "TikTok", font=ImageFont.load_default(size=22), fill=245)
    glyphs = np.asarray(lettering)
    reference = np.repeat(glyphs[:, :, None], 3, axis=2)
    context = ~ndimage.binary_dilation(glyphs > 0, iterations=4)
    rng = np.random.default_rng(12)
    moved = reference.copy()
    moved[context] = rng.integers(0, 255, (int(context.sum()), 3), dtype=np.uint8)
    assert scanner._present(moved, reference)
    without_logo = moved.copy()
    without_logo[~context] = 0
    assert not scanner._present(without_logo, reference)


def test_creator_component_and_fade_keep_exact_visible_start_stop(monkeypatch):
    frames = [np.zeros((160, 240, 3), np.uint8) for _ in range(12)]
    for index, frame in enumerate(frames):
        frame[0, 0, 0] = index
        if index < 5:
            frame[50:62:2, 20:80] = 245
            frame[51:62:2, 20:80:4] = 245
        if index < 8:
            intensity = 245 if index < 5 else (130, 90, 60)[index - 5]
            frame[70:80:3, 24:87] = intensity
            frame[71:80:3, 25:87:5] = intensity
        else:
            # An unrelated bright caption in the same general area must not
            # extend the old verified creator's disappearance time.
            frame[73:76, 24:87] = 245
    info = SimpleNamespace(
        width=240,
        height=160,
        total_frames=12,
        fps=8,
        duration=1.5,
        frame_times=[i / 8 for i in range(12)],
        scene_boundaries=[0, 12],
    )
    monkeypatch.setattr(scanner, "probe_video", lambda path: info)
    monkeypatch.setattr(scanner, "iter_frames", lambda path: iter(frames))
    reader = Reader(
        lambda index: [
            _text("TikTok", [20 / 240, 50 / 160, 80 / 240, 62 / 160]),
            _text("@examplecreator", [24 / 240, 70 / 160, 87 / 240, 80 / 160]),
        ]
        if index == 0
        else []
    )
    result = scanner.scan_video(Path("unused"), reader=reader)
    assert [(r["start_frame"], r["end_frame"]) for r in result["regions"]] == [(0, 8)]
    assert result["regions"][0]["timing_method"] == "verified-component-templates-v2"


def test_profiles_include_instagram_without_accepting_other_platforms(video):
    reader = Reader(lambda index: [_text("Instagram", [20 / 240, 50 / 160, 60 / 240, 60 / 160])])
    result = scanner.scan_video(Path("unused"), reader=reader, watermark_style="instagram")
    assert result["regions"] and result["regions"][0]["label"] == "instagram"
    assert result["scan_version"] == scanner.SCAN_VERSION == "platform-templates-v1"
    assert scanner.scan_video(Path("unused"), reader=reader, watermark_style="auto")["regions"]
    assert not scanner.scan_video(Path("unused"), reader=reader, watermark_style="tiktok")[
        "regions"
    ]
    with pytest.raises(ValueError, match="watermark style"):
        scanner.scan_video(Path("unused"), reader=reader, watermark_style="invalid")


def test_frame_verification_keeps_moving_backgrounds_and_genuine_disappearances(monkeypatch):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=19)
    lettering = Image.new("L", (80, 22))
    ImageDraw.Draw(lettering).text((0, 0), "TikTok", font=font, fill=255)
    glyphs = np.asarray(lettering) > 150
    rng = np.random.default_rng(41)
    frames = []
    for index in range(36):
        frame = rng.integers(0, 140, (200, 240, 3), dtype=np.uint8)
        frame[0, 0, 0] = index
        if index < 17:
            frame[50:72, 20:100][glyphs] = 245
        else:
            # Different white scene lettering replaces the old branding.
            frame[55:58, 20:100] = 245
        if index >= 17 and index not in (27, 28):
            frame[110:132, 150:230][glyphs] = 245
        frames.append(frame)
    info = SimpleNamespace(
        width=240,
        height=200,
        total_frames=len(frames),
        fps=12,
        duration=3,
        frame_times=[i / 12 for i in range(len(frames))],
        scene_boundaries=[0, len(frames)],
    )
    monkeypatch.setattr(scanner, "probe_video", lambda path: info)
    monkeypatch.setattr(scanner, "iter_frames", lambda path: iter(frames))
    reader = Reader(
        lambda index: [_text("TikTok", [20 / 240, 50 / 200, 100 / 240, 72 / 200])]
        if index == 0
        else [_text("TikTok", [150 / 240, 110 / 200, 230 / 240, 132 / 200])]
        if index == 18
        else []
    )
    result = scanner.scan_video(Path("unused"), reader=reader)
    assert [(r["start_frame"], r["end_frame"]) for r in result["regions"]] == [
        (0, 17),
        (17, 27),
        (29, 36),
    ]


def test_instagram_creator_requires_independent_adjacent_icon_and_allowed_style(video, monkeypatch):
    creator = _text("@examplecreator", [20 / 240, 50 / 160, 60 / 240, 60 / 160])
    reader = Reader(lambda index: [creator] if index == 0 else [])
    monkeypatch.setattr(scanner, "match_icon", lambda *args, **kwargs: None)
    assert not scanner.scan_video(Path("unused"), reader=reader)["regions"]

    calls = []

    def icon(frame, anchor, label, *, username=False):
        calls.append((anchor, label, username))
        return {"bbox": (8, 48, 18, 62), "score": 0.99, "template": "instagram-camera"}

    monkeypatch.setattr(scanner, "match_icon", icon)
    result = scanner.scan_video(Path("unused"), reader=reader, watermark_style="instagram")
    assert [(r["label"], r["start_frame"], r["end_frame"]) for r in result["regions"]] == [
        ("instagram", 0, 5),
        ("instagram", 9, 13),
    ]
    assert result["regions"][0]["icon_search"] == [6 / 240, 46 / 160, 20 / 240, 64 / 160]
    assert all(label == "instagram" and username for _, label, username in calls)
    assert not scanner.scan_video(Path("unused"), reader=reader, watermark_style="tiktok")[
        "regions"
    ]
    assert not scanner.scan_video(Path("unused"), "Sora", reader=reader)["regions"]


def test_username_in_scene_center_is_not_instagram_branding(video, monkeypatch):
    monkeypatch.setattr(
        scanner,
        "match_icon",
        lambda *args, **kwargs: {"bbox": (60, 80, 70, 90), "score": 0.99, "template": "camera"},
    )
    reader = Reader(lambda _: [_text("@examplecreator", [0.4, 0.4, 0.6, 0.45])])
    assert not scanner.scan_video(Path("unused"), reader=reader)["regions"]


def test_matching_platform_icon_replaces_fixed_ocr_anchor_geometry(monkeypatch):
    anchor = _text("TikTok", [41 / 468, 326 / 702, 123 / 468, 350 / 702])
    monkeypatch.setattr(
        scanner,
        "match_icon",
        lambda *args, **kwargs: {"bbox": (17, 322, 42, 353), "score": 0.98, "template": "tiktok"},
    )
    candidate = scanner._candidate(
        anchor, [anchor], 468, 702, "tiktok", frame=np.zeros((702, 468, 3), dtype=np.uint8)
    )
    assert candidate["icon"] == (15, 320, 44, 355)
    assert candidate["bbox"][0] <= 15


def test_creator_beside_live_tiktok_mark_is_not_reclassified_as_instagram(video, monkeypatch):
    monkeypatch.setattr(
        scanner,
        "match_icon",
        lambda *args, **kwargs: {"bbox": (8, 48, 18, 62), "score": 0.99, "template": "camera"},
    )
    creator = _text("@examplecreator", [20 / 240, 65 / 160, 60 / 240, 75 / 160])
    branding = _text("TikTok", [20 / 240, 50 / 160, 60 / 240, 60 / 160])
    # Handle is first in OCR order. Later OCR samples miss the wordmark, but its
    # independently confirmed shape still identifies the adjacent platform.
    reader = Reader(
        lambda index: [creator, branding]
        if index == 0
        else [creator]
        if index < 5 or 9 <= index < 13
        else []
    )
    result = scanner.scan_video(Path("unused"), reader=reader)
    assert [(r["label"], r["start_frame"], r["end_frame"]) for r in result["regions"]] == [
        ("tiktok", 0, 5),
        ("tiktok", 9, 13),
    ]


def test_matched_platform_icon_retains_faded_first_frame_without_padding(monkeypatch):
    from PIL import Image, ImageDraw

    camera = Image.new("L", (22, 22))
    draw = ImageDraw.Draw(camera)
    draw.rounded_rectangle((2, 2, 19, 19), radius=4, outline=255, width=3)
    draw.ellipse((7, 7, 14, 14), outline=255, width=2)
    icon_ink = np.asarray(camera) > 0
    frames = [np.zeros((160, 300, 3), np.uint8) for _ in range(12)]
    for index, frame in enumerate(frames):
        frame[0, 0, 0] = index
        if index < 8:
            intensity = 64 if index in (0, 7) else 245
            frame[48:70, 20:42][icon_ink] = intensity
        else:
            frame[54:57, 20:42] = 245  # Unrelated white scene detail replaces the icon.
        if 1 <= index < 7:
            frame[50:62:2, 50:120] = 245
            frame[51:62:2, 50:120:4] = 245
    info = SimpleNamespace(
        width=300,
        height=160,
        total_frames=len(frames),
        fps=8,
        duration=1.5,
        frame_times=[i / 8 for i in range(len(frames))],
        scene_boundaries=[0, len(frames)],
    )
    monkeypatch.setattr(scanner, "probe_video", lambda path: info)
    monkeypatch.setattr(scanner, "iter_frames", lambda path: iter(frames))
    monkeypatch.setattr(
        scanner,
        "match_icon",
        lambda *args, **kwargs: {
            "bbox": (20, 48, 42, 70),
            "score": 0.99,
            "template": "instagram",
        },
    )
    reader = Reader(
        lambda index: [_text("@examplecreator", [50 / 300, 50 / 160, 120 / 300, 62 / 160])]
        if index == 2
        else []
    )
    result = scanner.scan_video(Path("unused"), reader=reader)
    assert [(r["start_frame"], r["end_frame"]) for r in result["regions"]] == [(0, 8)]
    # Only independently matched icons join the text parts and establish timing.
    region = result["regions"][0]
    assert len(region["mask_parts"]) == 2
    assert region["icon_search"] in region["mask_parts"]


def test_geometric_icon_fallback_is_not_learned_as_verified_identity(monkeypatch):
    monkeypatch.setattr(scanner, "match_icon", lambda *args, **kwargs: None)
    anchor = _text("TikTok", [0.4, 0.5, 0.6, 0.55])
    candidate = scanner._candidate(
        anchor, [anchor], 240, 160, "tiktok", frame=np.zeros((160, 240, 3), np.uint8)
    )
    assert not candidate["icon_verified"]
    track = {"components": []}
    scanner._remember_components(track, candidate, np.zeros((160, 240, 3), np.uint8))
    assert [component["box"] for component in track["components"]] == candidate["parts"]


def test_later_icon_matches_replace_unverified_search_with_genuine_union(video, monkeypatch):
    reader = Reader(
        lambda index: [_text("TikTok", [20 / 240, 50 / 160, 60 / 240, 60 / 160])]
        if index in (0, 2, 4)
        else []
    )

    def icon(frame, anchor, label, *, username=False):
        index = int(frame[0, 0, 0])
        if index == 0:
            return None
        box = (8, 48, 18, 62) if index == 2 else (7, 48, 18, 63)
        return {"bbox": box, "score": 0.99, "template": "tiktok"}

    monkeypatch.setattr(scanner, "match_icon", icon)
    result = scanner.scan_video(Path("unused"), reader=reader)
    for region in result["regions"]:
        assert region["icon_search"] == [5 / 240, 46 / 160, 20 / 240, 65 / 160]
        assert region["icon_search"] in region["mask_parts"]


@pytest.mark.parametrize("style", ["instagram", "tiktok"])
def test_custom_creator_target_uses_selected_platform_only_with_matching_icon(
    video, monkeypatch, style
):
    reader = Reader(
        lambda index: [_text("@examplecreator", [20 / 240, 50 / 160, 60 / 240, 60 / 160])]
        if index == 0
        else []
    )
    monkeypatch.setattr(
        scanner,
        "match_icon",
        lambda *args, **kwargs: {"bbox": (8, 48, 18, 62), "score": 0.99, "template": style},
    )
    result = scanner.scan_video(
        Path("unused"), "examplecreator", reader=reader, watermark_style=style
    )
    assert {region["label"] for region in result["regions"]} == {style}
    assert result["regions"][0]["icon_search"] == [6 / 240, 46 / 160, 20 / 240, 64 / 160]

    monkeypatch.setattr(scanner, "match_icon", lambda *args, **kwargs: None)
    result = scanner.scan_video(
        Path("unused"), "examplecreator", reader=reader, watermark_style=style
    )
    assert {region["label"] for region in result["regions"]} == {"examplecreator"}


def test_custom_creator_uses_adjacent_wordmark_and_keeps_it_when_later_ocr_misses(
    video, monkeypatch
):
    creator = _text("@examplecreator", [20 / 240, 65 / 160, 85 / 240, 75 / 160])
    brand = _text("TikTok", [20 / 240, 50 / 160, 60 / 240, 60 / 160])
    for index, frame in enumerate(video):
        if index < 5 or 9 <= index < 13:
            frame[65:75:2, 20:85] = 245
            frame[66:75:2, 20:85:4] = 245
    reader = Reader(
        lambda index: [creator, {**brand, "text": "ТікТок"}]
        if index == 0
        else [creator, brand]
        if index == 2
        else [creator]
        if index < 5 or 9 <= index < 13
        else []
    )

    def icon(frame, anchor, label, *, username=False):
        # A faint note can be localized once TikTok has been independently read,
        # while being insufficient to infer a platform from a handle alone.
        if username:
            return None
        return {"bbox": (8, 48, 18, 62), "score": 0.79, "template": "tiktok"}

    monkeypatch.setattr(scanner, "match_icon", icon)
    result = scanner.scan_video(
        Path("unused"), "examplecreator", reader=reader, watermark_style="tiktok"
    )
    assert [(r["label"], r["start_frame"], r["end_frame"]) for r in result["regions"]] == [
        ("tiktok", 0, 5),
        ("tiktok", 9, 13),
    ]
    for region in result["regions"]:
        assert region["anchor_bbox"] == brand["bbox"]
        assert creator["bbox"] in region["mask_parts"]
        assert region["icon_search"] in region["mask_parts"]


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        (") @examplecreator", True),
        ("hello @examplecreator", False),
        ("@examplecreator is here", False),
        (")))))))@examplecreator", False),
    ],
)
def test_instagram_handle_allows_only_limited_leading_ocr_punctuation(
    video, monkeypatch, text, accepted
):
    calls = []

    def icon(frame, anchor, label, *, username=False):
        calls.append(username)
        return {"bbox": (8, 48, 18, 62), "score": 0.99, "template": "reels"}

    monkeypatch.setattr(scanner, "match_icon", icon)
    reader = Reader(
        lambda index: [_text(text, [20 / 240, 50 / 160, 60 / 240, 60 / 160])] if index == 0 else []
    )
    result = scanner.scan_video(Path("unused"), reader=reader, watermark_style="instagram")
    assert bool(result["regions"]) is accepted
    if accepted:
        assert all(calls)  # Punctuation never lowers the independent icon threshold.
        assert {r["label"] for r in result["regions"]} == {"instagram"}
