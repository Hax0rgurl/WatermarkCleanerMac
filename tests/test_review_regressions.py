"""Regression cases retained after replacing the original restoration pipeline."""

from datetime import datetime
import subprocess
from types import SimpleNamespace

import pytest

from tests.media_fixtures import create_video
from watermark_cleaner import core, video
from watermark_cleaner.completion import SourceIdentityError
from watermark_cleaner.timed_regions import compile_regions


class IdentityEngine:
    def restore(self, frames, masks, progress=None):
        return frames.copy()


def region(end=20):
    return [{"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": end - 1, "end_frame": end}]


@pytest.mark.parametrize("value", [None, [], {}, "bad"])
def test_invalid_coordinate_has_clear_value_error(value):
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        compile_regions([[value, 0.1, 0.3, 0.3]], 120, 80, 20)


def test_last_decoded_frame_of_variable_rate_video_is_selectable(tmp_path):
    source = create_video(tmp_path / "input.mp4", pts=[0, 3000, 12000, 15000, 21000])
    info = video.probe_video(source)
    assert info.total_frames == 5
    assert info.frame_times == [0, 1 / 30, 2 / 15, 1 / 6, 7 / 30]
    assert compile_regions(region(5), 120, 80, info.total_frames)[0].active(4)
    output = tmp_path / "cleaned.mp4"
    core.WatermarkCleaner(engine=IdentityEngine()).run(source, output, watermark_regions=region(5))
    video.verify_output(source, output)


@pytest.mark.parametrize("stage", ["mux", "verify"])
def test_finishing_timeout_keeps_source_and_does_not_publish(tmp_path, monkeypatch, stage):
    source = create_video(tmp_path / "input.mp4", audio=True)
    original = source.read_bytes()
    output = tmp_path / "cleaned.mp4"
    workspace = tmp_path / "job"
    workspace.mkdir()
    sentinel = workspace / "user-file.txt"
    sentinel.write_text("keep")
    actual_run = subprocess.run

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 3600
        is_verification = "-xerror" in command
        if is_verification == (stage == "verify"):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return actual_run(command, **kwargs)

    monkeypatch.setattr(video.subprocess, "run", timeout)
    expected_error = RuntimeError if stage == "mux" else subprocess.TimeoutExpired
    with pytest.raises(expected_error) as failure:
        core.WatermarkCleaner(engine=IdentityEngine()).run(
            source, output, watermark_regions=region(), working_directory=workspace
        )
    if stage == "mux":
        assert "Finishing the video timed out after 3600 seconds" in str(failure.value)
        assert isinstance(failure.value.__cause__, subprocess.TimeoutExpired)
    assert source.read_bytes() == original
    assert not output.exists()
    assert list(workspace.iterdir()) == [sentinel]


def test_unique_output_names_do_not_hang_when_the_clock_is_frozen(tmp_path, monkeypatch):
    now = datetime(2026, 10, 26, 5, 30, 0, 123456)
    monkeypatch.setattr(core, "datetime", SimpleNamespace(now=lambda: now))
    first = core.unique_mp4_path(tmp_path, "source")
    first.write_bytes(b"keep")
    second = core.unique_mp4_path(tmp_path, "source")
    second.symlink_to(tmp_path / "missing")
    third = core.unique_mp4_path(tmp_path, "source")
    assert first.name == "source-cleaned-20261026-053000-123456.mp4"
    assert second.name == "source-cleaned-20261026-053000-123456-2.mp4"
    assert third.name == "source-cleaned-20261026-053000-123456-3.mp4"
    assert first.read_bytes() == b"keep" and second.is_symlink()


@pytest.mark.parametrize("collision", ["source", "existing", "symlink", "hardlink"])
def test_colliding_output_cannot_replace_any_file(tmp_path, collision):
    source = create_video(tmp_path / "input.mp4")
    original = source.read_bytes()
    output = tmp_path / "cleaned.mp4"
    if collision == "source":
        output = source
    elif collision == "existing":
        output.write_bytes(b"existing user file")
    elif collision == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    with pytest.raises((ValueError, FileExistsError)):
        core.WatermarkCleaner(engine=IdentityEngine()).run(
            source, output, watermark_regions=region()
        )
    assert source.read_bytes() == original
    if collision == "existing":
        assert output.read_bytes() == b"existing user file"


@pytest.mark.parametrize(
    "metadata",
    [
        {"anchor_bbox": None},
        {"anchor_bbox": [0.1, 0.1, 0.1, 0.5]},
        {"anchor_bbox": [float("nan"), 0.1, 0.3, 0.5]},
        {"icon_search": [-0.1, 0.1, 0.3, 0.5]},
        {"mask_parts": []},
        {"mask_parts": [[0.1, 0.1, 1.5, 0.5]]},
    ],
)
def test_untrusted_optional_region_metadata_is_validated_before_model_loading(
    tmp_path, monkeypatch, metadata
):
    source = create_video(tmp_path / "input.mp4")
    monkeypatch.setattr(
        core, "get_engine", lambda *args: pytest.fail("Invalid metadata loaded the model")
    )
    selection = [{**region()[0], **metadata}]
    with pytest.raises(ValueError):
        core.WatermarkCleaner().run(source, tmp_path / "out.mp4", watermark_regions=selection)
    assert not (tmp_path / "out.mp4").exists()


def test_source_symlink_cannot_be_used_as_an_identity_pinned_input(tmp_path):
    source = create_video(tmp_path / "input.mp4")
    link = tmp_path / "source-link.mp4"
    link.symlink_to(source)
    original = source.read_bytes()
    with pytest.raises(SourceIdentityError, match="symbolic link"):
        core.WatermarkCleaner(engine=IdentityEngine()).run(
            link, tmp_path / "out.mp4", watermark_regions=region()
        )
    assert link.is_symlink() and source.read_bytes() == original


def test_decoder_cannot_end_before_the_probed_frame_count(tmp_path, monkeypatch):
    source = create_video(tmp_path / "input.mp4")
    actual = video.iter_frames

    def short_frames(path):
        for index, frame in enumerate(actual(path)):
            if index == 10:
                return
            yield frame

    monkeypatch.setattr(core, "iter_frames", short_frames)
    with pytest.raises(ValueError, match="ended before"):
        core.WatermarkCleaner(engine=IdentityEngine()).run(
            source, tmp_path / "out.mp4", watermark_regions=region()
        )
    assert source.exists() and not (tmp_path / "out.mp4").exists()
