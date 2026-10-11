"""Regression coverage for fixes ported from upstream PR 11."""
import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from demark_world import completion, core
from demark_world.schemas import CleanerType
from demark_world.timed_regions import compile_regions
from demark_world.utils.video_utils import VideoLoader


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Create a real, small video with twenty decoded frames."""
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 80))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((80, 120, 3), 50, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("value", [None, [], {}, "bad"])
def test_invalid_coordinate_has_clear_value_error(value: object) -> None:
    """Malformed coordinates follow the public validation contract."""
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        compile_regions([[value, .1, .3, .3]], 120, 80, 20)


@pytest.mark.parametrize("reported", [0, 10, 100, float("nan")])
def test_decoded_count_ignores_metadata(video: Path, monkeypatch: pytest.MonkeyPatch, reported: float) -> None:
    """Even the final real frame remains selectable with inaccurate metadata."""
    real = cv2.VideoCapture
    class Capture:
        def __init__(self, path: str) -> None:
            self.capture = real(path)
        def get(self, prop: int) -> float:
            return reported if prop == cv2.CAP_PROP_FRAME_COUNT else self.capture.get(prop)
        def isOpened(self) -> bool:
            return self.capture.isOpened()
        def read(self):
            return self.capture.read()
        def release(self) -> None:
            self.capture.release()
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    loader = VideoLoader(video)
    assert loader.total_frames == 20
    assert compile_regions([{"bbox": [.1,.1,.3,.3], "start_frame": 19, "end_frame": 20}],120,80,loader.total_frames)[0].active(19)


def test_real_public_api_finishes_and_verifies(video: Path, tmp_path: Path) -> None:
    """Run encoding, optional audio fallback, and complete output decode."""
    class Cleaner:
        def clean(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            return image
    output = tmp_path / "cleaned.mp4"
    original = hashlib.sha256(video.read_bytes()).hexdigest()
    engine = core.DeMarkWorld(cleaner_type=CleanerType.LAMA, automatic_detection=False, cleaner=Cleaner())
    engine.run(video, output, quiet=True, watermark_regions=[{"bbox": [.1,.1,.3,.3], "start_frame": 19, "end_frame": 20}])
    completion.verify_cleaned_video(output)
    assert VideoLoader(output).total_frames == 20
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original


def test_mux_timeout_keeps_source_and_does_not_publish(video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A finishing timeout cannot publish partial media or consume the source."""
    partial = tmp_path / "video.partial.mp4"
    partial.write_bytes(b"partial")
    output = tmp_path / "output.mp4"
    original = video.read_bytes()
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 3600
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(core.subprocess, "run", timeout)
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=object())
    with pytest.raises(subprocess.TimeoutExpired):
        engine.merge_audio_track(video, partial, output)
    assert video.read_bytes() == original and not output.exists()
    assert not list(tmp_path.glob("*.audio-partial.mp4"))


def test_verification_timeout_is_a_failed_verification(tmp_path: Path) -> None:
    """A verifier timeout must never authorize source retirement."""
    output = tmp_path / "output.mp4"
    output.write_bytes(b"fixture")
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 3600
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    with pytest.raises(completion.OutputVerificationError):
        completion.verify_cleaned_video(output, runner=timeout, ffmpeg_executable="ffmpeg")
