import pytest

from demark_world.watermark_guard import (
    matches_target, target_names, verified_detections, anchor_model_detections,
)
from demark_world.timed_regions import compile_regions


def test_specific_identity_is_required():
    for value in ["", " ", "AI", ","]:
        with pytest.raises(ValueError):
            target_names(value)
    assert target_names("InVideo, @muse") == ["invideo", "muse"]


def test_identity_matches_whole_names_not_unrelated_words():
    assert matches_target("invideo/muse", ["invideo"])
    assert matches_target("IN VIDEO / muse", ["invideo"])
    assert matches_target("@Muse", ["muse"])
    assert not matches_target("Soraya", ["sora"])
    assert not matches_target("1 FOCUS", ["invideo", "sora"])


def test_copied_logo_descriptions_do_not_make_visible_brands_unfindable():
    names = target_names(":fish: InVideo/muse, S Storyblocks, iStock by Getty Images")
    assert names == ["invideo", "storyblocks", "istock", "getty images"]
    assert matches_target("invideo/muse", names)
    assert matches_target("Storyblocks", names)
    assert matches_target("iStock", names)
    assert matches_target("Getty Images", names)
    assert not matches_target("FOCUS", names)


def test_model_claiming_white_caption_is_invideo_is_rejected():
    model = [{"label":"InVideo", "bbox":[.04,.65,.2,.85]}]
    ocr = [{"text":"1 FOCUS", "confidence":.99, "bbox":[.05,.7,.17,.84]}]
    assert verified_detections(model, ocr, ["invideo"]) == []


def test_brand_elsewhere_does_not_authorize_a_caption_box():
    caption = {"label":"Watermark", "bbox":[.04,.65,.2,.85]}
    logo = {"label":"InVideo", "bbox":[.76,.09,.96,.17]}
    ocr = [{"text":"invideo/muse", "confidence":.95, "bbox":[.78,.10,.94,.16]},
           {"text":"FOCUS", "confidence":.99, "bbox":[.05,.7,.17,.84]}]
    assert verified_detections([caption,logo], ocr, ["invideo"]) == [logo]


def test_giant_text_boxes_and_unreadable_identity_are_rejected():
    large = {"label":"InVideo", "bbox":[.1,.1,.95,.95]}
    small = {"label":"InVideo", "bbox":[.76,.09,.96,.17]}
    ocr = [{"text":"invideo", "confidence":.95, "bbox":[.78,.10,.94,.16]}]
    assert verified_detections([large], ocr, ["invideo"]) == []
    ocr[0]["confidence"] = .3
    assert verified_detections([small], ocr, ["invideo"]) == []


def test_old_unchecked_suggestions_cannot_reach_the_cleaner():
    area = {"bbox":[.1,.1,.2,.2], "start_frame":0, "end_frame":30,
            "source":"suggested", "label":"FOCUS"}
    with pytest.raises(ValueError, match="identity check"):
        compile_regions([area], 100,100,30)
    area["verification"] = "named-ocr-v1"
    assert len(compile_regions([area],100,100,30)) == 1


def test_manual_box_still_works_for_a_symbol_without_text():
    area = {"bbox":[.1,.1,.2,.2], "start_frame":0, "end_frame":30,
            "source":"manual"}
    assert len(compile_regions([area],100,100,30)) == 1


def test_model_can_confirm_brand_when_its_box_cuts_off_some_lettering():
    model = [{"label": "invideo/muse", "bbox": [.79,.10,.92,.13]}]
    ocr = [{"text": "invideo/muse", "confidence": .99, "bbox": [.81,.11,.95,.15]}]
    assert verified_detections(model, ocr, ["invideo"]) == []
    completed = anchor_model_detections(model, ocr, ["invideo"], 1280, 720)
    assert len(completed) == 1
    assert completed[0]["bbox"][2] == .95
    assert completed[0]["bbox"][3] >= .15


def test_completing_model_box_cannot_reach_unrelated_caption_or_distant_brand():
    model = [{"label": "InVideo", "bbox": [.1,.7,.4,.8]}]
    ocr = [{"text": "FOCUS", "confidence": .99, "bbox": [.1,.7,.4,.8]},
           {"text": "invideo/muse", "confidence": .99, "bbox": [.81,.11,.95,.15]}]
    assert anchor_model_detections(model, ocr, ["invideo"], 1280, 720) == []
