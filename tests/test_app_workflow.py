"""Desktop UI contracts, with restoration itself tested by the engine integration suite."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
from watermark_cleaner import completion


def _run_app():
    import app

    app.main()


@pytest.fixture
def desktop_ui(monkeypatch, tmp_path):
    """Keep all native selections, outputs and job receipts inside this test."""
    output = tmp_path / "output"
    output.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    working = tmp_path / "working"
    quarantine = tmp_path / "failed"
    monkeypatch.setattr(app, "OUTPUT_DIR", output)
    monkeypatch.setattr(app, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(app, "LOGS_PATH", logs)
    monkeypatch.setattr(app, "_configure_logging", lambda: 0)
    monkeypatch.setattr(app, "_model_status", lambda: {"e2fgvi": True, "propainter": True})
    monkeypatch.setattr(app, "regions_for_processing", lambda identity, **kwargs: None)
    monkeypatch.setattr(
        app,
        "create_job_workspace",
        lambda identity, destination: completion.create_job_workspace(
            identity, destination, working_root=working
        ),
    )
    monkeypatch.setattr(
        app,
        "quarantine_recorded_output",
        lambda workspace, **kwargs: completion.quarantine_recorded_output(
            workspace, quarantine_root=quarantine, audit_path=logs / "audit.jsonl", **kwargs
        ),
    )
    monkeypatch.setattr(
        app,
        "quarantine_job_workspace",
        lambda workspace: completion.quarantine_job_workspace(
            workspace, quarantine_root=quarantine, audit_path=logs / "audit.jsonl"
        ),
    )
    monkeypatch.setattr(
        app,
        "complete_job_workspace",
        lambda workspace: completion.complete_job_workspace(
            workspace, audit_path=logs / "audit.jsonl"
        ),
    )
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"version": 1, "sources": []}))
    monkeypatch.setenv("MAC_WATERMARK_SELECTION_FILE", str(manifest))
    return {"output": output, "manifest": manifest, "quarantine": quarantine}


def _queue(desktop_ui, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"unit-test-source-identity")
    identity = completion.SourceIdentity.capture(source)
    desktop_ui["manifest"].write_text(json.dumps({"version": 1, "sources": [asdict(identity)]}))
    return source


def test_two_engines_default_to_e2fgvi_without_loading_a_model(desktop_ui, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_new_cleaner", lambda engine: calls.append(engine))
    ui = AppTest.from_function(_run_app).run()
    assert not ui.exception
    assert ui.selectbox(key="restoration-engine").value == "E2FGVI-HQ"
    assert ui.selectbox(key="restoration-engine").options == ["E2FGVI-HQ", "ProPainter"]
    assert ui.selectbox(key="watermark-style").value == "Auto"
    assert ui.selectbox(key="watermark-style").options == [
        "Auto",
        "TikTok",
        "Instagram",
        "CapCut",
        "InVideo",
    ]
    assert ui.button(key="remove-watermarks").disabled
    assert not calls


def test_first_launch_prepares_both_models_once(desktop_ui, monkeypatch):
    ready = False
    calls = []

    def status():
        return {"e2fgvi": ready, "propainter": ready}

    def prepare(progress):
        nonlocal ready
        calls.append("both")
        progress(0.5, "Downloading local models")
        ready = True

    monkeypatch.setattr(app, "_model_status", status)
    monkeypatch.setattr(app, "_download_models", prepare)
    ui = AppTest.from_function(_run_app).run()
    assert not ui.exception
    assert calls == ["both"]
    ui.run()
    assert calls == ["both"]
    assert not ui.error


def test_failed_setup_keeps_processing_disabled_and_has_retry(desktop_ui, monkeypatch):
    attempts = []
    monkeypatch.setattr(app, "_model_status", lambda: {"e2fgvi": False, "propainter": False})

    def offline(progress):
        attempts.append(1)
        raise OSError("Download unavailable while offline")

    monkeypatch.setattr(app, "_download_models", offline)
    ui = AppTest.from_function(_run_app).run()
    assert not ui.exception
    assert ui.button(key="remove-watermarks").disabled
    assert any("offline" in error.value for error in ui.error)
    ui.selectbox(key="restoration-engine").select("ProPainter").run()
    assert len(attempts) == 1
    ui.button(key="retry-model-setup").click().run()
    assert len(attempts) == 2
    assert ui.button(key="remove-watermarks").disabled


def test_chosen_engine_processes_and_keeps_original(desktop_ui, tmp_path, monkeypatch):
    source = _queue(desktop_ui, tmp_path)
    before = source.read_bytes()
    chosen = []
    calls = []

    class VerifiedCleaner:
        def run(self, input_path, output_path, **kwargs):
            calls.append(kwargs)
            kwargs["progress_callback"](35)
            partial = kwargs["working_directory"] / "verified.partial.mp4"
            partial.write_bytes(b"pipeline-test-output")
            kwargs["before_publish"](partial, output_path)
            os.replace(partial, output_path)
            kwargs["progress_callback"](100)

    def factory(engine_id):
        chosen.append(engine_id)
        return VerifiedCleaner()

    monkeypatch.setattr(app, "_new_cleaner", factory)
    ui = AppTest.from_function(_run_app).run()
    ui.selectbox(key="restoration-engine").select("ProPainter").run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception
    assert not ui.error
    assert chosen == ["propainter"]
    assert source.read_bytes() == before
    assert calls[0]["watermark_regions"] is None
    assert calls[0]["watermark_text"] == ""
    assert calls[0]["watermark_style"] == "auto"
    assert len(list(desktop_ui["output"].glob("*.mp4"))) == 1
    assert any("Original kept" in item.value for item in ui.success)
    assert len(ui.get("download_button")) == 1


def test_selected_watermark_style_reaches_detection_and_restoration(
    desktop_ui, tmp_path, monkeypatch
):
    source = _queue(desktop_ui, tmp_path)
    scans, runs = [], []
    regions = [{"bbox": [0.1, 0.1, 0.2, 0.2], "source": "manual"}]

    def find(identity, **kwargs):
        assert identity.source_path == source
        scans.append(kwargs)
        kwargs["progress"](0.5, "Checking the chosen style")
        return regions

    class Cleaner:
        def run(self, input_path, output_path, **kwargs):
            runs.append(kwargs)
            partial = kwargs["working_directory"] / "verified.partial.mp4"
            partial.write_bytes(b"pipeline-test-output")
            kwargs["before_publish"](partial, output_path)
            os.replace(partial, output_path)

    monkeypatch.setattr(app, "regions_for_processing", find)
    monkeypatch.setattr(app, "_new_cleaner", lambda engine: Cleaner())
    ui = AppTest.from_function(_run_app).run()
    ui.selectbox(key="watermark-style").select("Instagram").run()
    ui.text_input(key="watermark-name").input("creator").run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception and not ui.error
    assert len(scans) == len(runs) == 1
    assert scans[0]["watermark_style"] == runs[0]["watermark_style"] == "instagram"
    assert scans[0]["watermark_text"] == runs[0]["watermark_text"] == "creator"
    assert runs[0]["watermark_regions"] == regions


def test_failed_automatic_refresh_cannot_process_or_publish(desktop_ui, tmp_path, monkeypatch):
    source = _queue(desktop_ui, tmp_path)
    original = source.read_bytes()
    calls = []

    class Cleaner:
        def run(self, *args, **kwargs):
            calls.append(kwargs)

    def unavailable(identity, **kwargs):
        raise RuntimeError("Watermark timing could not be checked")

    monkeypatch.setattr(app, "regions_for_processing", unavailable)
    monkeypatch.setattr(app, "_new_cleaner", lambda engine: Cleaner())
    ui = AppTest.from_function(_run_app).run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception
    assert any("Watermark timing could not be checked" in item.value for item in ui.error)
    assert source.read_bytes() == original
    assert not calls and not list(desktop_ui["output"].glob("*.mp4"))
    assert not ui.session_state["completed"] and not ui.get("download_button")
    assert list(desktop_ui["quarantine"].rglob("job.json"))


def test_source_is_revalidated_after_automatic_refresh(desktop_ui, tmp_path, monkeypatch):
    source = _queue(desktop_ui, tmp_path)
    calls = []

    class Cleaner:
        def run(self, *args, **kwargs):
            calls.append(kwargs)

    def swapped(identity, **kwargs):
        replacement = source.with_name("replacement.mp4")
        replacement.write_bytes(b"replacement-at-the-selected-path")
        os.replace(replacement, source)
        return [{"bbox": [0.1, 0.1, 0.2, 0.2], "source": "manual"}]

    monkeypatch.setattr(app, "regions_for_processing", swapped)
    monkeypatch.setattr(app, "_new_cleaner", lambda engine: Cleaner())
    ui = AppTest.from_function(_run_app).run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception and ui.error
    assert not calls and not list(desktop_ui["output"].glob("*.mp4"))
    assert not ui.session_state["completed"] and not ui.get("download_button")
    assert list(desktop_ui["quarantine"].rglob("job.json"))


def test_failed_processing_has_no_completed_download_and_keeps_source(
    desktop_ui, tmp_path, monkeypatch
):
    source = _queue(desktop_ui, tmp_path)
    before = source.read_bytes()

    class FailingCleaner:
        def run(self, input_path, output_path, **kwargs):
            (kwargs["working_directory"] / "failed.partial.mp4").write_bytes(b"partial")
            raise RuntimeError("Video restoration failed")

    monkeypatch.setattr(app, "_new_cleaner", lambda engine: FailingCleaner())
    ui = AppTest.from_function(_run_app).run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception
    assert any("Video restoration failed" in item.value for item in ui.error)
    assert source.read_bytes() == before
    assert not list(desktop_ui["output"].glob("*.mp4"))
    assert not ui.session_state["completed"]
    assert not ui.get("download_button")
    assert list(desktop_ui["quarantine"].rglob("failed.partial.mp4"))


def test_replaced_source_cannot_publish_a_result(desktop_ui, tmp_path, monkeypatch):
    source = _queue(desktop_ui, tmp_path)

    class SwappedSourceCleaner:
        def run(self, input_path, output_path, **kwargs):
            replacement = source.with_name("replacement.mp4")
            replacement.write_bytes(b"different-source-at-same-path")
            os.replace(replacement, source)
            partial = kwargs["working_directory"] / "verified.partial.mp4"
            partial.write_bytes(b"pipeline-test-output")
            kwargs["before_publish"](partial, output_path)
            os.replace(partial, output_path)

    monkeypatch.setattr(app, "_new_cleaner", lambda engine: SwappedSourceCleaner())
    ui = AppTest.from_function(_run_app).run()
    ui.button(key="remove-watermarks").click().run()
    assert not ui.exception
    assert ui.error
    assert not list(desktop_ui["output"].glob("*.mp4"))
    assert not ui.session_state["completed"]


def test_visual_timeline_maps_variable_frame_times_exactly():
    """Check the actual browser functions against a variable-rate frame sequence."""
    html = (
        Path(app.__file__).parent / "watermark_cleaner/watermark_selector/index.html"
    ).read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    timing = re.search(r"function frameTime.*?(?=function selectedRegion)", script, re.S)
    assert timing is not None
    bundled_node = Path(
        "/Users/muse/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    )
    node = shutil.which("node") or (str(bundled_node) if bundled_node.is_file() else None)
    assert node, "Node is needed to validate the browser timeline"
    check = "const vm=require('node:vm');new vm.Script(" + json.dumps(script) + ");\n"
    check += "let fps=30,total=4,frameTimes=[0,1/30,.1,.2],duration=.25;\n" + timing.group()
    check += "const actual=[frameTime(2),frameAt(.099),frameAt(.1),frameAt(.15),frameAt(.15,true),frameTime(4),frameAt(.25,true),frameAt(.25)];if(JSON.stringify(actual)!==JSON.stringify([.1,1,2,2,3,.25,4,3]))throw new Error(JSON.stringify(actual));"
    publish = re.search(r"function publish.*?(?=function change)", script, re.S)
    assert publish is not None
    check += "let sent,sourceKey='scope-a',revision=7,regions=[];function controls(){}function send(type,payload){sent=payload;}"
    check += publish.group()
    check += "publish();if(sent.value.source_key!==sourceKey||sent.value.revision!==revision)throw new Error('Visual edits lost their scope');"
    mutation = re.search(r"function change.*?(?=function point)", script, re.S)
    assert mutation is not None
    check += "let corrected={source:'suggested',verification:'named-ocr-v1',timing_method:'verified-component-templates-v2',timing_verified:true,mask_parts:[[.1,.1,.2,.2]],anchor_bbox:[.1,.1,.2,.2],icon_search:[.1,.1,.2,.2]};function selectedRegion(){return corrected};"
    check += mutation.group()
    check += "change(r=>r.start_frame=2);if(corrected.source!=='manual'||corrected.start_frame!==2||['verification','timing_method','timing_verified','mask_parts','anchor_bbox','icon_search'].some(k=>k in corrected))throw new Error('Manual correction retained an old automatic mask');"
    subprocess.run([node, "-e", check], check=True, capture_output=True, text=True)


def test_saved_manual_ranges_survive_restart_and_respect_watermark_scope(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    saved, summary = selector.selection_files(identity)
    regions = [
        {
            "id": "manual",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "start_frame": 1,
            "end_frame": 3,
            "source": "manual",
            "label": "TikTok",
        }
    ]
    selector._save(saved, regions)
    selector._save(
        summary,
        {
            "schema": 3,
            "target": "tiktok",
            "total_frames": 4,
            "edited": True,
            "timing_verified": False,
        },
    )
    restored = selector.selected_regions(identity, "TikTok")
    assert restored and restored[0]["start_frame"] == 1 and restored[0]["end_frame"] == 3
    assert selector.selected_regions(identity, "InVideo") is None
    assert selector.selected_regions(identity, "") is None
    monkeypatch.setattr(
        scanner,
        "scan_video",
        lambda *args, **kwargs: {
            "total_frames": 4,
            "regions": [
                {
                    **regions[0],
                    "id": "fresh-auto",
                    "source": "suggested",
                    "verification": "named-ocr-v1",
                    "timing_verified": True,
                    "mask_parts": [regions[0]["bbox"]],
                }
            ],
        },
    )
    selector.regions_for_processing(identity)
    monkeypatch.setattr(selector.st, "session_state", {})
    assert selector.selected_regions(identity, "TikTok") == restored


def test_removed_all_manual_ranges_does_not_silently_rescan(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    state = {}
    monkeypatch.setattr(selector.st, "session_state", state)
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    key = selector.selection_key(identity)
    state[f"{key}-areas"] = []
    state[f"{key}-scan-result"] = {"schema": 3, "target": "", "total_frames": 4}
    state[f"{key}-edited"] = True
    with pytest.raises(ValueError, match="No watermark ranges"):
        selector.selected_regions(identity)


def test_selector_preserves_valid_scanner_mask_metadata():
    from watermark_cleaner import region_selector as selector

    regions = [
        {
            "id": "scan",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "start_frame": 0,
            "end_frame": 4,
            "source": "suggested",
            "label": "TikTok",
            "verification": "named-ocr-v1",
            "timing_verified": True,
            "mask_parts": [[0.11, 0.11, 0.19, 0.15]],
            "anchor_bbox": [0.11, 0.11, 0.19, 0.15],
            "icon_search": [0.10, 0.10, 0.12, 0.12],
        }
    ]
    assert all(
        selector._normalized(regions, 4)[0][key] == regions[0][key]
        for key in ("mask_parts", "anchor_bbox", "icon_search")
    )
    regions[0]["end_frame"] = 8
    with pytest.raises(ValueError):
        selector._normalized(regions, 4)


def test_visual_preview_retains_real_variable_frame_timestamps(tmp_path):
    import base64
    from fractions import Fraction
    from io import BytesIO

    import av
    import numpy as np
    from PIL import Image
    from watermark_cleaner import region_selector as selector

    path = tmp_path / "variable-rate.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 48, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 30)
        for index, pts in enumerate([0, 1, 3, 5]):
            pixels = np.zeros((48, 32, 3), dtype=np.uint8)
            pixels[:, :, index % 3] = 200
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    preview = selector.video_preview(str(path), path.stat().st_mtime_ns)
    assert preview["total_frames"] == 4
    assert preview["frame_times"] == [0, 1 / 30, 3 / 30, 5 / 30]
    image = Image.open(BytesIO(base64.b64decode(preview["image"].split(",", 1)[1])))
    assert image.size == (32, 48)
    red, green, blue = image.getpixel((16, 24))
    assert red > 150 and green < 30 and blue < 30


def test_stale_automatic_ranges_refresh_without_replacing_manual_corrections(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    manual = {
        "id": "user-corrected",
        "bbox": [0.1, 0.1, 0.3, 0.2],
        "start_frame": 5,
        "end_frame": 7,
        "source": "manual",
        "label": "TikTok",
    }

    def automatic(identifier, box, start, stop, *, with_parts=True):
        region = {
            "id": identifier,
            "bbox": box,
            "start_frame": start,
            "end_frame": stop,
            "source": "suggested",
            "label": "TikTok",
            "verification": "named-ocr-v1",
            "timing_verified": True,
        }
        if with_parts:
            region["mask_parts"] = [box]
        return region

    previous = automatic("old-uncorrected", [0.6, 0.1, 0.8, 0.2], 20, 30, with_parts=False)
    corrected = automatic("old-boundary", manual["bbox"], 3, 10)
    deleted = automatic("removed-by-user", [0.4, 0.4, 0.5, 0.5], 10, 15)
    saved, summary_file = selector.selection_files(identity)
    selector._save(saved, [manual, previous])
    selector._save(
        summary_file,
        {
            "schema": 3,
            "target": "",
            "total_frames": 40,
            "timing_verified": True,
            "edited": True,
            "overrides": [corrected, deleted],
        },
    )
    calls = []
    updated = automatic("fresh-uncorrected", previous["bbox"], 19, 31)
    added = automatic("newly-found", [0.05, 0.8, 0.15, 0.9], 32, 37)

    def scan(path, **kwargs):
        calls.append(kwargs)
        assert path == source
        return {"regions": [corrected, deleted, updated, added], "total_frames": 40}

    monkeypatch.setattr(scanner, "scan_video", scan)
    assert selector.selected_regions(identity) is None
    result = selector.regions_for_processing(identity)
    assert [region["id"] for region in result] == [
        "user-corrected",
        "fresh-uncorrected",
        "newly-found",
    ]
    assert result[0] == manual
    assert result[1]["mask_parts"] == updated["mask_parts"]
    assert calls[0]["watermark_style"] == "auto"
    assert selector.regions_for_processing(identity) == result
    assert len(calls) == 1
    monkeypatch.setattr(selector.st, "session_state", {})
    assert selector.selected_regions(identity) == result
    summary = json.loads(selector.selection_files(identity, "", "auto")[1].read_text())
    assert summary["scan_version"] == scanner.SCAN_VERSION
    assert summary["overrides"] == [corrected, deleted]


@pytest.mark.parametrize("correction", [False, True])
def test_refreshing_fragmented_timing_retains_coverage_outside_manual_override(
    tmp_path, monkeypatch, correction
):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)

    def automatic(identifier, first, last):
        return {
            "id": identifier,
            "bbox": [0.1, 0.1, 0.3, 0.2],
            "start_frame": first,
            "end_frame": last,
            "source": "suggested",
            "label": "TikTok",
            "verification": "named-ocr-v1",
            "timing_verified": True,
            "mask_parts": [[0.1, 0.1, 0.3, 0.2]],
        }

    old_fragment = automatic("old-fragment", 70, 74 if correction else 71)
    manual = (
        [
            {
                **old_fragment,
                "id": "manual-correction",
                "start_frame": 71,
                "end_frame": 73,
                "source": "manual",
            }
        ]
        if correction
        else []
    )
    saved, summary = selector.selection_files(identity)
    selector._save(saved, [automatic("old-first", 0, 65), *manual, automatic("old-last", 74, 150)])
    selector._save(
        summary,
        {
            "schema": 3,
            "target": "",
            "total_frames": 150,
            "timing_verified": True,
            "edited": True,
            "overrides": [old_fragment],
        },
    )
    monkeypatch.setattr(
        scanner,
        "scan_video",
        lambda *args, **kwargs: {
            "regions": [automatic("fresh-continuous", 0, 150)],
            "total_frames": 150,
        },
    )
    result = selector.regions_for_processing(identity)
    expected = (
        [(0, 70, "suggested"), (71, 73, "manual"), (74, 150, "suggested")]
        if correction
        else [(0, 70, "suggested"), (71, 150, "suggested")]
    )
    assert [(r["start_frame"], r["end_frame"], r["source"]) for r in result] == expected
    assert len({region["id"] for region in result}) == len(result)
    assert selector.regions_for_processing(identity) == result
    monkeypatch.setattr(selector.st, "session_state", {})
    assert selector.selected_regions(identity) == result


@pytest.mark.parametrize("legacy", [False, True])
def test_added_manual_box_is_additive_on_refresh_and_legacy_corrections_remain(
    tmp_path, monkeypatch, legacy
):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    automatic = {
        "id": "automatic-wordmark",
        "bbox": [0.08, 0.1, 0.3, 0.2],
        "start_frame": 0,
        "end_frame": 150,
        "source": "suggested",
        "label": "TikTok",
        "verification": "named-ocr-v1",
        "timing_verified": True,
        "mask_parts": [[0.1, 0.1, 0.3, 0.2]],
    }
    manual = {
        "id": "added-missing-icon",
        "bbox": [0.08, 0.1, 0.12, 0.2],
        "start_frame": 70,
        "end_frame": 74,
        "source": "manual",
        "label": "Watermark",
    }
    summary = {
        "schema": 3,
        "target": "",
        "total_frames": 150,
        "edited": True,
    }
    if not legacy:
        summary["overrides"] = []
    saved, summary_file = selector.selection_files(identity)
    selector._save(saved, [automatic, manual])
    selector._save(summary_file, summary)
    monkeypatch.setattr(
        scanner,
        "scan_video",
        lambda *args, **kwargs: {"regions": [automatic], "total_frames": 150},
    )
    result = selector.regions_for_processing(identity)
    expected = (
        [(0, 70, "suggested"), (70, 74, "manual"), (74, 150, "suggested")]
        if legacy
        else [(0, 150, "suggested"), (70, 74, "manual")]
    )
    assert [(r["start_frame"], r["end_frame"], r["source"]) for r in result] == expected
    stored = json.loads(selector.selection_files(identity, "", "auto")[1].read_text())
    assert stored["overrides"] == ([manual] if legacy else [])
    key = selector.selection_key(identity)
    selector.st.session_state[f"{key}-scan-result"]["scan_version"] = "previous-scanner"
    assert selector.regions_for_processing(identity) == result
    monkeypatch.setattr(selector.st, "session_state", {})
    assert selector.selected_regions(identity) == result


def test_extended_manual_correction_overrides_fresh_geometry_for_its_whole_interval(
    tmp_path, monkeypatch
):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    automatic = {
        "id": "original-auto-fragment",
        "bbox": [0.08, 0.1, 0.3, 0.2],
        "start_frame": 70,
        "end_frame": 74,
        "source": "suggested",
        "label": "TikTok",
        "verification": "named-ocr-v1",
        "timing_verified": True,
        "mask_parts": [[0.1, 0.1, 0.3, 0.2]],
    }
    manual = {
        "id": automatic["id"],
        "bbox": [0.12, 0.12, 0.2, 0.18],
        "start_frame": 60,
        "end_frame": 90,
        "source": "manual",
        "label": "TikTok",
    }
    saved, summary_file = selector.selection_files(identity)
    selector._save(
        saved,
        [manual, {**automatic, "id": "stale-auto", "start_frame": 90, "end_frame": 150}],
    )
    selector._save(
        summary_file,
        {
            "schema": 3,
            "target": "",
            "total_frames": 150,
            "edited": True,
            "overrides": [automatic],
        },
    )
    monkeypatch.setattr(
        scanner,
        "scan_video",
        lambda *args, **kwargs: {
            "regions": [
                {**automatic, "id": "fresh-continuous", "start_frame": 0, "end_frame": 150}
            ],
            "total_frames": 150,
        },
    )
    key = selector.selection_key(identity)
    result = selector.regions_for_processing(identity)
    assert [(r["start_frame"], r["end_frame"], r["source"]) for r in result] == [
        (0, 60, "suggested"),
        (60, 90, "manual"),
        (90, 150, "suggested"),
    ]
    assert result[1] == manual
    selector.st.session_state[f"{key}-scan-result"]["scan_version"] = "previous-scanner"
    assert selector.regions_for_processing(identity) == result
    monkeypatch.setattr(selector.st, "session_state", {})
    assert selector.selected_regions(identity) == result


def test_watermark_styles_keep_separate_manual_selections(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(selector.st, "session_state", {})
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    for style, start in (("tiktok", 1), ("instagram", 5)):
        regions = [
            {
                "id": style,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "start_frame": start,
                "end_frame": start + 2,
                "source": "manual",
                "label": style,
            }
        ]
        saved, summary_file = selector.selection_files(identity, watermark_style=style)
        selector._save(saved, regions)
        selector._save(
            summary_file,
            {"schema": 3, "target": "", "style": style, "total_frames": 10, "edited": True},
        )
    assert selector.selected_regions(identity) is None
    assert selector.selected_regions(identity, watermark_style="tiktok")[0]["start_frame"] == 1
    assert selector.selected_regions(identity, watermark_style="instagram")[0]["start_frame"] == 5
    assert selector.selected_regions(identity, watermark_style="tiktok")[0]["start_frame"] == 1


def test_visual_editor_scopes_responses_and_keeps_manual_boundary_override(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector, scanner

    monkeypatch.setattr(selector, "RUNTIME_ROOT", tmp_path)
    state = {}
    monkeypatch.setattr(selector.st, "session_state", state)
    monkeypatch.setattr(selector.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(selector.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(selector.st, "info", lambda *args, **kwargs: None)
    source = tmp_path / "selected.mp4"
    source.write_bytes(b"source-identity-only")
    identity = completion.SourceIdentity.capture(source)
    monkeypatch.setattr(
        selector,
        "video_preview",
        lambda *args: {
            "image": "",
            "fps": 10,
            "total_frames": 10,
            "frame_times": [],
            "duration": 1,
        },
    )

    class Preview:
        def register(self, identity):
            return "http://127.0.0.1/local-test"

    monkeypatch.setattr(selector, "_preview_server", Preview)
    regions = [
        {
            "id": "wrong-style",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "start_frame": 0,
            "end_frame": 5,
            "source": "manual",
        }
    ]
    monkeypatch.setattr(
        selector,
        "_select_regions",
        lambda **kwargs: {
            "source_key": selector.selection_key(identity, watermark_style="tiktok"),
            "revision": kwargs["revision"],
            "regions": regions,
        },
    )
    assert selector.select_watermark_regions(identity, watermark_style="instagram") == []
    assert not selector.selection_files(identity, watermark_style="instagram")[0].exists()
    monkeypatch.setattr(
        selector,
        "_select_regions",
        lambda **kwargs: {
            "source_key": kwargs["source_key"],
            "revision": kwargs["revision"],
            "regions": regions,
        },
    )
    accepted = selector.select_watermark_regions(identity, watermark_style="instagram")
    assert accepted[0]["id"] == "wrong-style"
    assert selector.selected_regions(identity, watermark_style="instagram") == accepted
    assert selector.selected_regions(identity, watermark_style="tiktok") is None
    key = selector.selection_key(identity, watermark_style="instagram")
    automatic = {
        **accepted[0],
        "source": "suggested",
        "verification": "named-ocr-v1",
        "timing_verified": True,
        "mask_parts": [accepted[0]["bbox"]],
    }
    state[f"{key}-areas"] = [automatic]
    state[f"{key}-scan-result"]["scan_version"] = scanner.SCAN_VERSION
    regions = [{**accepted[0], "start_frame": 1, "end_frame": 4}]
    corrected = selector.select_watermark_regions(identity, watermark_style="instagram")
    assert corrected[0]["start_frame"] == 1 and corrected[0]["end_frame"] == 4
    assert state[f"{key}-scan-result"]["overrides"] == [automatic]
