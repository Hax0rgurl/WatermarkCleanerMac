from __future__ import annotations

import importlib
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


def test_demark_imports_do_not_load_mmcv() -> None:
    import demark_world.watermark_cleaner  # noqa: F401

    assert "mmcv" not in sys.modules
    assert "mmengine" not in sys.modules


def test_runtime_paths_follow_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAC_WATERMARK_RUNTIME_DIR", str(tmp_path / "runtime"))
    sys.modules.pop("demark_world.configs", None)
    configs = importlib.import_module("demark_world.configs")

    assert configs.RUNTIME_ROOT == tmp_path / "runtime"
    assert configs.WATER_MARK_DETECT_YOLO_WEIGHTS.is_relative_to(configs.RUNTIME_ROOT)
    assert configs.LOGS_PATH.is_relative_to(configs.RUNTIME_ROOT)
    assert configs.DATA_PATH.is_relative_to(configs.RUNTIME_ROOT)


def test_legacy_runtime_environment_is_still_supported(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MAC_WATERMARK_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("SORA_WATERMARK_RUNTIME_DIR", str(tmp_path / "legacy-runtime"))
    sys.modules.pop("demark_world.configs", None)
    configs = importlib.import_module("demark_world.configs")

    assert configs.RUNTIME_ROOT == tmp_path / "legacy-runtime"


def test_engine_accepts_injected_components() -> None:
    from demark_world.core import DeMarkWorld
    from demark_world.schemas import CleanerType

    detector = object()
    cleaner = object()
    engine = DeMarkWorld(
        cleaner_type=CleanerType.LAMA,
        detector=detector,
        cleaner=cleaner,
        detect_batch_size=2,
    )
    assert engine.detector is detector
    assert engine.cleaner is cleaner


def test_invalid_detection_batch_size_is_rejected() -> None:
    from demark_world.core import DeMarkWorld

    with pytest.raises(ValueError, match="at least 1"):
        DeMarkWorld(detector=object(), cleaner=object(), detect_batch_size=0)


def test_overlap_blending_contract() -> None:
    from demark_world.utils.video_utils import merge_frames_with_overlap

    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    result = merge_frames_with_overlap(None, [black, black], 0, 1, True)
    result = merge_frames_with_overlap(result, [white, white], 1, 1)
    assert len(result) == 3
    assert np.array_equal(result[0], black)
    assert np.array_equal(result[1], black)
    assert np.array_equal(result[2], white)


def test_abort_encoding_terminates_child_and_removes_partial_file(tmp_path: Path) -> None:
    from demark_world.core import DeMarkWorld

    class FakeInput:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeInput()
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            return -15

        def kill(self) -> None:
            self.terminated = True

    partial = tmp_path / "temp-output.mp4"
    partial.write_bytes(b"partial")
    process = FakeProcess()

    DeMarkWorld._abort_encoding(process, partial)

    assert process.stdin.closed
    assert process.terminated
    assert not partial.exists()


def test_progress_failure_aborts_outer_pipeline(monkeypatch, tmp_path: Path) -> None:
    import demark_world.core as core
    from demark_world.schemas import CleanerType

    class FakeLoader:
        width = 2
        height = 2
        fps = 1.0
        total_frames = 1
        original_bitrate = None

        def __init__(self, path: Path) -> None:
            self.path = path

        def __iter__(self):
            yield np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeInput:
        closed = False

        def write(self, data: bytes) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeInput()
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            return -15

        def kill(self) -> None:
            self.terminated = True

    process = FakeProcess()

    class FakeGraph:
        def __init__(self) -> None:
            self.output_path = None

        def output(self, output_path: str, **kwargs):
            self.output_path = Path(output_path)
            return self

        def overwrite_output(self):
            return self

        def global_args(self, *args):
            return self

        def run_async(self, **kwargs):
            self.output_path.write_bytes(b"partial")
            return process

    class FakeDetector:
        def detect_batch(self, frames, batch_size):
            return [
                {
                    "detected": False,
                    "bbox": None,
                    "confidence": None,
                    "center": None,
                }
                for _ in frames
            ]

    monkeypatch.setattr(core, "VideoLoader", FakeLoader)
    monkeypatch.setattr(core.ffmpeg, "input", lambda *args, **kwargs: FakeGraph())
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")

    engine = core.DeMarkWorld(
        cleaner_type=CleanerType.LAMA,
        detector=FakeDetector(),
        cleaner=object(),
    )

    with pytest.raises(RuntimeError, match="callback failure"):
        engine.run(
            input_path,
            output_path,
            progress_callback=lambda _: (_ for _ in ()).throw(
                RuntimeError("callback failure")
            ),
        )

    assert process.terminated
    assert not output_path.exists()
    assert not (tmp_path / ".output.partial.mp4").exists()


def test_preexisting_output_is_never_modified(tmp_path: Path) -> None:
    from demark_world.core import DeMarkWorld

    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "existing.mp4"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"keep-me")
    engine = DeMarkWorld(detector=object(), cleaner=object())

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        engine.run(input_path, output_path)

    assert output_path.read_bytes() == b"keep-me"


def test_no_detections_never_publish_an_unchanged_video(
    monkeypatch, tmp_path: Path
) -> None:
    import demark_world.core as core
    from demark_world.schemas import CleanerType

    frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
    ]

    class FakeLoader:
        width = 2
        height = 2
        fps = 2.0
        total_frames = 2
        original_bitrate = None

        def __init__(self, path: Path) -> None:
            pass

        def __iter__(self):
            yield from frames

    class FakeInput:
        def __init__(self) -> None:
            self.closed = False
            self.data = bytearray()

        def write(self, data: bytes) -> None:
            self.data.extend(data)

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeInput()
            self.output_path = None

        def poll(self):
            return 0 if self.stdin.closed else None

        def wait(self, timeout=None) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    process = FakeProcess()

    class FakeGraph:
        def output(self, output_path: str, **kwargs):
            process.output_path = Path(output_path)
            return self

        def overwrite_output(self):
            return self

        def global_args(self, *args):
            return self

        def run_async(self, **kwargs):
            process.output_path.write_bytes(b"encoded")
            return process

    class EmptyDetector:
        def detect_batch(self, batch, batch_size):
            return [
                {
                    "detected": False,
                    "bbox": None,
                    "confidence": None,
                    "center": None,
                }
                for _ in batch
            ]

    class CleanerMustNotRun:
        def clean(self, frame, mask):
            raise AssertionError("cleaner must not run without detections")

    monkeypatch.setattr(core, "VideoLoader", FakeLoader)
    monkeypatch.setattr(core.ffmpeg, "input", lambda *args, **kwargs: FakeGraph())
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")
    engine = core.DeMarkWorld(
        cleaner_type=CleanerType.LAMA,
        detector=EmptyDetector(),
        cleaner=CleanerMustNotRun(),
    )
    monkeypatch.setattr(
        engine,
        "merge_audio_track",
        lambda _, temporary, final, working_directory=None, before_publish=None: (
            engine._publish_no_clobber(
                temporary,
                final,
                before_publish=before_publish,
            )
        ),
    )
    workspace = tmp_path / "Working" / "job-test"

    with pytest.raises(ValueError, match="Automatic detection found no watermark"):
        engine.run(
            input_path,
            output_path,
            quiet=True,
            working_directory=workspace,
        )

    assert not output_path.exists()
    assert input_path.read_bytes() == b"video"
    assert process.output_path == workspace / "video.partial.mp4"
    assert not process.output_path.exists()
    assert not process.stdin.data


class _FakeDownloadResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        if self.payload:
            yield self.payload


@pytest.mark.parametrize("payload", [b"", b"truncated"])
def test_bad_detector_download_is_rejected(
    monkeypatch, tmp_path: Path, payload: bytes
) -> None:
    from demark_world.utils import download_utils

    expected = hashlib.sha256(b"correct model").hexdigest()
    model_path = tmp_path / "detector.pt"
    monkeypatch.setattr(
        download_utils.requests,
        "get",
        lambda *args, **kwargs: _FakeDownloadResponse(payload),
    )

    with pytest.raises(RuntimeError, match="could not be downloaded"):
        download_utils.ensure_model_downloaded(
            model_path,
            "https://example.invalid/detector.pt",
            expected_sha256=expected,
        )

    assert not model_path.exists()
    assert not model_path.with_suffix(".pt.download").exists()


def test_corrupt_cached_detector_is_revalidated_and_replaced(
    monkeypatch, tmp_path: Path
) -> None:
    from demark_world.utils import download_utils

    correct = b"correct model"
    expected = hashlib.sha256(correct).hexdigest()
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"corrupt cache")
    calls = 0

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _FakeDownloadResponse(correct)

    monkeypatch.setattr(download_utils.requests, "get", fake_get)
    download_utils.ensure_model_downloaded(
        model_path,
        "https://example.invalid/detector.pt",
        expected_sha256=expected,
    )

    assert calls == 1
    assert model_path.read_bytes() == correct


def test_detector_digest_matches_current_github_release_asset() -> None:
    from demark_world.configs import WATER_MARK_DETECT_YOLO_WEIGHTS_SHA256

    assert (
        WATER_MARK_DETECT_YOLO_WEIGHTS_SHA256
        == "6ac71b6ab8db27ec7928b5176e60a359"
        "c65e1579a5c1d58cf2f98df30cf3085e"
    )
