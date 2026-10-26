"""Selection boundaries remain exact on the temporal restoration pipeline."""

import pytest

from watermark_cleaner.timed_regions import compile_regions, normalize_regions, region_boxes


def test_legacy_visual_rectangles_cover_every_decoded_frame():
    region = compile_regions([[0.1, 0.1, 0.2, 0.2]], 100, 100, 30)[0]
    assert region.bbox == (10, 10, 20, 20)
    assert region.start_frame == 0
    assert region.end_frame == 30
    assert region.active(0) and region.active(29) and not region.active(30)


@pytest.mark.parametrize(
    "start,end", [(-1, 10), (5, 5), (10, 9), (0, 31), (1.5, 10), (True, 10), (0, False)]
)
def test_invalid_frame_ranges_are_rejected(start, end):
    with pytest.raises(ValueError):
        normalize_regions(
            [{"bbox": [0.1, 0.1, 0.2, 0.2], "start_frame": start, "end_frame": end}], 30
        )


def test_multiple_timed_boxes_are_independently_end_exclusive():
    regions = compile_regions(
        [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 2, "end_frame": 4},
            {"bbox": [0.7, 0.6, 0.9, 0.9], "start_frame": 5, "end_frame": 7},
        ],
        120,
        80,
        8,
    )
    assert [[region.active(index) for index in range(8)] for region in regions] == [
        [False, False, True, True, False, False, False, False],
        [False, False, False, False, False, True, True, False],
    ]


def test_automatic_regions_require_independent_identity_marker():
    region = {"bbox": [0.1, 0.1, 0.3, 0.3], "source": "suggested", "start_frame": 0, "end_frame": 5}
    with pytest.raises(ValueError, match="identity"):
        compile_regions([region], 100, 80, 10)
    assert compile_regions([{**region, "verification": "named-ocr-v1"}], 100, 80, 10)


def test_normalization_preserves_reviewed_timing_and_labels():
    result = normalize_regions(
        [
            {
                "id": "mine",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "label": "TikTok",
                "source": "suggested",
                "verification": "named-ocr-v1",
                "timing_verified": True,
                "start_frame": 4,
                "end_frame": 9,
            }
        ],
        10,
    )[0]
    assert result["id"] == "mine"
    assert result["label"] == "TikTok"
    assert result["timing_verified"]
    assert result["verification"] == "named-ocr-v1"


def test_subpixel_boxes_enclose_every_selected_pixel():
    assert region_boxes([[0.011, 0.012, 0.019, 0.029]], 100, 100) == [(1, 1, 2, 3)]
