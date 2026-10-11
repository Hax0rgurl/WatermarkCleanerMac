import io
from types import SimpleNamespace

import numpy as np
import pytest

from demark_world.timed_regions import compile_regions, normalize_regions
from demark_world.vision_scan import (
    local_endpoint, parse_watermarks, ranges_from_observations, refine_ranges, sample_indices,
    verify_range_boundaries, recurring_cadence,
)


def test_legacy_boxes_cover_the_entire_video():
    region = compile_regions([[.1, .1, .2, .2]], 100, 100, 30)[0]
    assert region.start_frame == 0 and region.end_frame == 30
    assert region.active(0) and region.active(29) and not region.active(30)


@pytest.mark.parametrize("start,end", [(-1, 10), (5, 5), (10, 9), (0, 31),
                                      (1.5, 10), (True, 10)])
def test_invalid_ranges_are_rejected(start, end):
    with pytest.raises(ValueError):
        normalize_regions([{"bbox": [.1,.1,.2,.2], "start_frame": start,
                            "end_frame": end}], 30)


def test_cleaning_applies_each_box_only_to_its_own_frames(monkeypatch, tmp_path):
    import demark_world.core as core
    frames = [np.full((80, 120, 3), 50, np.uint8) for _ in range(8)]
    class Cleaner:
        def __init__(self):
            self.calls = 0
        def clean(self, image, mask):
            self.calls += 1
            output = image.copy()
            output[mask > 0] = 200
            return output
    cleaner = Cleaner()
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=cleaner)
    monkeypatch.setattr(core, "VideoLoader", lambda _: frames)
    sink = SimpleNamespace(stdin=io.BytesIO())
    regions = compile_regions([
        {"bbox": [.1,.1,.3,.3], "start_frame": 2, "end_frame": 4},
        {"bbox": [.7,.6,.9,.9], "start_frame": 5, "end_frame": 7},
    ], 120, 80, 8)
    engine._clean_lama(tmp_path/"input.mp4", [None]*8, 120, 80, 8,
                       sink, None, True, timed_regions=regions)
    output = np.frombuffer(sink.stdin.getvalue(), np.uint8).reshape(8,80,120,3)
    assert cleaner.calls == 4
    for index in [0,1,4,7]:
        assert np.array_equal(output[index], frames[index])
    for index in [2,3]:
        assert np.all(output[index, 12:20, 16:30] == 200)
        assert np.all(output[index, 55:60, 88:96] == 50)
    for index in [5,6]:
        assert np.all(output[index, 55:60, 88:96] == 200)
        assert np.all(output[index, 12:20, 16:30] == 50)


def test_scan_splits_disappearance_and_handles_two_simultaneous_logos():
    a = {"label":"A", "bbox":[.1,.1,.2,.2]}
    b = {"label":"B", "bbox":[.7,.7,.8,.8]}
    observations = [{"frame":0,"watermarks":[a,b]},
                    {"frame":10,"watermarks":[a]},
                    {"frame":20,"watermarks":[]},
                    {"frame":30,"watermarks":[a]},
                    {"frame":39,"watermarks":[a]}]
    regions = ranges_from_observations(observations, 40)
    assert [(r["label"],r["start_frame"],r["end_frame"]) for r in regions] == [
        ("A",0,15), ("B",0,5), ("A",25,40)]


def test_sampling_always_includes_the_last_frame():
    assert sample_indices(101, 30, 1) == [0,30,60,90,100]


def test_vision_boundary_refinement_finds_first_and_last_active_frames():
    logo = {"label":"A", "bbox":[.1,.1,.2,.2]}
    observations = [{"frame":0,"watermarks":[]}, {"frame":10,"watermarks":[logo]},
                    {"frame":20,"watermarks":[]}]
    regions = ranges_from_observations(observations, 21)
    refined = refine_ranges(regions, observations, lambda f: [logo] if 7<=f<16 else [])
    assert refined[0]["start_frame"] == 7
    assert refined[0]["end_frame"] == 16


def test_vision_json_is_validated_and_never_silently_becomes_no_detection():
    assert parse_watermarks('```json\n{"watermarks":[]}\n```') == []
    assert parse_watermarks('{"watermarks":[{"label":"InVideo", "bbox":[770,90,950,160]}]}')[0]["bbox"] == [.77,.09,.95,.16]
    for answer in ['Sorry', '{"watermarks":[', '{"watermarks":[{"bbox":[0,0,1100,500]}]}']:
        with pytest.raises(ValueError):
            parse_watermarks(answer)


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://example.com/v1",
                                "http://localhost.evil/v1", "file:///tmp/test"])
def test_vision_frames_cannot_be_sent_to_an_external_server(url):
    with pytest.raises(ValueError):
        local_endpoint(url)


def test_local_server_url():
    assert local_endpoint("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080/v1"


def test_boundary_audit_rejects_a_logo_remaining_outside_the_selected_time():
    logo = {"label": "InVideo", "bbox": [.7,.1,.9,.2]}
    region = {"bbox": logo["bbox"], "start_frame": 10, "end_frame": 20}
    with pytest.raises(ValueError, match="boundary"):
        verify_range_boundaries([region], 40, lambda f: [logo] if 9 <= f < 20 else [])
    assert verify_range_boundaries([region], 40, lambda f: [logo] if 10 <= f < 20 else []) == 4
    assert region["timing_verified"]


def test_periodic_watermark_is_discovered_between_old_five_second_samples():
    logo = {"label": "InVideo", "bbox": [.7,.1,.9,.2]}
    def visible(f):
        return [logo] if any(start <= f < start+20 for start in [10,80,150,220]) else []
    observations = [{"frame": f, "watermarks": visible(f)}
                    for f in sample_indices(300, 10, 1)]
    regions = refine_ranges(ranges_from_observations(observations, 300), observations, visible)
    verify_range_boundaries(regions, 300, visible)
    assert [(r["start_frame"],r["end_frame"]) for r in regions] == [(10,30),(80,100),(150,170),(220,240)]
    assert recurring_cadence(regions, 10)["period_seconds"] == 7


def test_visual_confirmation_can_extend_past_an_ocr_miss_at_the_boundary():
    logo = {"label": "InVideo", "bbox": [.7,.1,.9,.2]}
    observations = [{"frame": f, "watermarks": [logo] if f == 30 else []}
                    for f in [0,10,20,30,40,50,60]]
    visible = lambda f: [logo] if 12 <= f < 48 else []
    regions = refine_ranges(ranges_from_observations(observations, 61), observations, visible)
    assert (regions[0]["start_frame"], regions[0]["end_frame"]) == (12,48)
    verify_range_boundaries(regions, 61, visible)
