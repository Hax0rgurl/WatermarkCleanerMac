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
    monkeypatch.setattr(app, "selected_regions", lambda identity, target: None)
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
    assert len(list(desktop_ui["output"].glob("*.mp4"))) == 1
    assert any("Original kept" in item.value for item in ui.success)
    assert len(ui.get("download_button")) == 1


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
    mutation = re.search(r"function change.*?(?=function point)", script, re.S)
    assert mutation is not None
    check += "let corrected={source:'suggested',timing_verified:true,mask_parts:[[.1,.1,.2,.2]],anchor_bbox:[.1,.1,.2,.2],icon_search:[.1,.1,.2,.2]};function selectedRegion(){return corrected}function publish(){};"
    check += mutation.group()
    check += "change(r=>r.start_frame=2);if(corrected.source!=='manual'||corrected.start_frame!==2||['timing_verified','mask_parts','anchor_bbox','icon_search'].some(k=>k in corrected))throw new Error('Manual correction retained an old automatic mask');"
    subprocess.run([node, "-e", check], check=True, capture_output=True, text=True)


def test_saved_manual_ranges_survive_restart_and_respect_watermark_scope(tmp_path, monkeypatch):
    from watermark_cleaner import region_selector as selector

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
