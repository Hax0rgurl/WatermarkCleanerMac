from demark_world.region_selector import scan_scope_matches


def test_changing_model_or_optional_server_keeps_the_checked_visible_timeline():
    summary = {"semantic_model_check": True, "options": {
        "watermark_text": ":fish: InVideo/muse, S Storyblocks, iStock by Getty Images",
        "sampling_seconds": 1, "profile": "Quick", "endpoint": "", "use_vision_model": True}}
    current = {**summary["options"], "profile": "Latest quality",
               "endpoint": "http://127.0.0.1:8080/v1", "server_model": "some-model"}
    assert scan_scope_matches(summary, current)
    assert not scan_scope_matches(summary, {**current, "watermark_text": "Sora"})
    assert not scan_scope_matches(summary, {**current, "sampling_seconds": .5})


def test_ocr_only_scan_is_not_reused_when_vision_confirmation_is_requested():
    summary = {"semantic_model_check": False, "options": {
        "watermark_text": "InVideo", "sampling_seconds": 1, "use_vision_model": False}}
    assert not scan_scope_matches(summary, {**summary["options"], "use_vision_model": True})
