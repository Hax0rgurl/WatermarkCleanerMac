"""Exercise the public pipeline with real local video I/O and controlled model outputs."""

import errno
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.media_fixtures import create_video
from watermark_cleaner import core
from watermark_cleaner.completion import SourceIdentity, SourceIdentityError
from watermark_cleaner.video import iter_frames, verify_output


class PaintEngine:
    def __init__(self):
        self.calls = []

    def restore(self, frames, masks, progress=None):
        self.calls.append((frames.copy(), masks.copy()))
        if progress:
            progress(0, "starting")
            progress(1, "finished")
        # A model must never be able to repaint the context outside the selection.
        return np.full(frames.shape, [20, 200, 40], dtype=np.uint8)


@pytest.fixture
def source(tmp_path):
    return create_video(tmp_path / "source.mp4")


@pytest.fixture
def recorded_writer(monkeypatch):
    writes = []
    original = core.VideoWriter

    class RecordingWriter(original):
        def write(self, image):
            writes.append(image.copy())
            return super().write(image)

    monkeypatch.setattr(core, "VideoWriter", RecordingWriter)
    return writes


def selected(start=0, end=20, bbox=None):
    return [{"bbox": bbox or [0.1, 0.1, 0.3, 0.3], "start_frame": start, "end_frame": end}]


def test_real_pipeline_only_changes_selected_pixels_and_times(source, tmp_path, recorded_writer):
    original = list(iter_frames(source))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "cleaned.mp4"
    engine = PaintEngine()
    regions = selected(2, 4) + selected(8, 10, [0.7, 0.6, 0.9, 0.9])
    core.WatermarkCleaner(engine=engine).run(source, output, watermark_regions=regions)
    verify_output(source, output)
    assert len(recorded_writer) == len(original) == 20
    for index, (before, after) in enumerate(zip(original, recorded_writer, strict=True)):
        mask = np.zeros(before.shape[:2], bool)
        if 2 <= index < 4:
            mask[8:24, 12:36] = True
        if 8 <= index < 10:
            mask[48:72, 84:108] = True
        assert np.array_equal(after[~mask], before[~mask])
        if mask.any():
            assert np.all(after[mask] == [20, 200, 40])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum


def test_overlap_windows_keep_every_frame_in_order(source, tmp_path, recorded_writer, monkeypatch):
    monkeypatch.setattr(core, "SOURCE_BUFFER_BUDGET_BYTES", 8 * 120 * 80 * 3)
    engine = PaintEngine()
    core.WatermarkCleaner(engine=engine).run(
        source, tmp_path / "out.mp4", watermark_regions=selected()
    )
    assert [len(frames) for frames, _ in engine.calls] == [8, 8, 8]
    assert len(recorded_writer) == 20
    # The context pixels expose dropped, repeated or incorrectly carried frames.
    original = list(iter_frames(source))
    assert [int(frame[0, 0, 0]) for frame in recorded_writer] == [
        int(frame[0, 0, 0]) for frame in original
    ]


def test_overlap_predictions_are_blended_without_repainting_context(
    source, tmp_path, recorded_writer, monkeypatch
):
    monkeypatch.setattr(core, "SOURCE_BUFFER_BUDGET_BYTES", 8 * 120 * 80 * 3)

    class DistinctWindows(PaintEngine):
        def restore(self, frames, masks, progress=None):
            self.calls.append((frames.copy(), masks.copy()))
            return np.full(frames.shape, 60 * len(self.calls), np.uint8)

    core.WatermarkCleaner(engine=DistinctWindows()).run(
        source, tmp_path / "out.mp4", watermark_regions=selected()
    )
    assert recorded_writer[5][12, 16, 0] == 60
    assert recorded_writer[6][12, 16, 0] == 80
    assert recorded_writer[7][12, 16, 0] == 100
    assert recorded_writer[8][12, 16, 0] == 120
    assert recorded_writer[12][12, 16, 0] == 140
    assert recorded_writer[13][12, 16, 0] == 160


def test_temporal_context_never_crosses_a_scene_cut(tmp_path):
    source = create_video(
        tmp_path / "cut.mp4",
        pts=[i * 3000 for i in range(10)],
        colors=[[25] * 3] * 5 + [[220] * 3] * 5,
    )
    engine = PaintEngine()
    core.WatermarkCleaner(engine=engine).run(
        source, tmp_path / "out.mp4", watermark_regions=selected(end=10)
    )
    assert [len(frames) for frames, _ in engine.calls] == [5, 5]
    assert all(frames.mean() < 50 or frames.mean() > 190 for frames, _ in engine.calls)


def test_regions_at_different_positions_get_same_shot_clean_reference_frames(source, tmp_path):
    engine = PaintEngine()
    regions = selected(0, 10) + selected(10, 20, [0.7, 0.6, 0.9, 0.9])
    core.WatermarkCleaner(engine=engine).run(
        source, tmp_path / "out.mp4", watermark_regions=regions
    )
    assert len(engine.calls) == 2
    assert all(len(frames) == 20 for frames, _ in engine.calls)
    assert engine.calls[0][1][:10].any() and not engine.calls[0][1][10:].any()
    assert not engine.calls[1][1][:10].any() and engine.calls[1][1][10:].any()


def test_explicit_empty_selection_never_silently_scans(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        core,
        "scan_video",
        lambda *args, **kwargs: pytest.fail("Empty manual selection triggered automatic scan"),
    )
    with pytest.raises(ValueError, match="watermark"):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, tmp_path / "out.mp4", watermark_regions=[]
        )


@pytest.mark.parametrize("engine_failure", ["raise", "shape", "dtype"])
def test_model_failure_never_publishes_or_deletes_source(
    source, tmp_path, monkeypatch, engine_failure
):
    original = source.read_bytes()
    workspace = tmp_path / "job"
    workspace.mkdir()
    sentinel = workspace / "user-file.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(core, "SOURCE_BUFFER_BUDGET_BYTES", 8 * 120 * 80 * 3)

    class Broken(PaintEngine):
        def restore(self, frames, masks, progress=None):
            self.calls.append(1)
            if engine_failure == "raise":
                if len(self.calls) == 2:
                    raise RuntimeError("model failure")
                return frames
            return frames[:-1] if engine_failure == "shape" else frames.astype(np.float32)

    with pytest.raises((ValueError, RuntimeError)):
        core.WatermarkCleaner(engine=Broken()).run(
            source, tmp_path / "out.mp4", watermark_regions=selected(), working_directory=workspace
        )
    assert not (tmp_path / "out.mp4").exists()
    assert source.read_bytes() == original
    assert list(workspace.iterdir()) == [sentinel]


def test_source_swap_during_processing_is_detected_before_mux(source, tmp_path, monkeypatch):
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(source.read_bytes())
    replacement_bytes = replacement.read_bytes()

    class Swap(PaintEngine):
        def restore(self, frames, masks, progress=None):
            replacement.replace(source)
            return frames

    monkeypatch.setattr(
        core, "mux_audio", lambda *args, **kwargs: pytest.fail("Source swap reached finishing mux")
    )
    with pytest.raises(SourceIdentityError):
        core.WatermarkCleaner(engine=Swap()).run(
            source, tmp_path / "out.mp4", watermark_regions=selected()
        )
    assert source.read_bytes() == replacement_bytes
    assert not (tmp_path / "out.mp4").exists()


def test_publication_receipt_names_verified_candidate_inode(source, tmp_path):
    output = tmp_path / "out.mp4"
    receipts = []

    def publication(partial, destination):
        assert partial != source and partial.is_file()
        assert destination == output
        verify_output(source, partial)
        receipts.append(SourceIdentity.capture(partial))

    core.WatermarkCleaner(engine=PaintEngine()).run(
        source, output, watermark_regions=selected(), before_publish=publication
    )
    assert output.stat().st_ino == receipts[0].inode
    assert output.stat().st_dev == receipts[0].device


def test_source_swap_in_publication_callback_cannot_publish(source, tmp_path):
    output = tmp_path / "out.mp4"
    replacement = tmp_path / "new-source.mp4"
    replacement.write_bytes(source.read_bytes())

    def publication(partial, destination):
        replacement.replace(source)

    with pytest.raises(SourceIdentityError):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, output, watermark_regions=selected(), before_publish=publication
        )
    assert source.is_file()
    assert not output.exists()


def test_verified_candidate_change_is_detected_before_publication(source, tmp_path):
    output = tmp_path / "out.mp4"

    def publication(partial, destination):
        with partial.open("ab") as handle:
            handle.write(b"changed after verification")

    with pytest.raises(SourceIdentityError):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, output, watermark_regions=selected(), before_publish=publication
        )
    assert not output.exists()
    assert source.is_file()


def test_output_collision_during_processing_preserves_foreign_file(source, tmp_path):
    output = tmp_path / "out.mp4"

    def publication(partial, destination):
        destination.write_bytes(b"another job's output")

    with pytest.raises(FileExistsError):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, output, watermark_regions=selected(), before_publish=publication
        )
    assert output.read_bytes() == b"another job's output"
    assert source.is_file()


def test_progress_failure_only_removes_our_unchanged_published_file(source, tmp_path):
    output = tmp_path / "out.mp4"

    def progress(value):
        if value == 100:
            raise RuntimeError("UI stopped")

    with pytest.raises(RuntimeError, match="UI stopped"):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, output, progress_callback=progress, watermark_regions=selected()
        )
    assert not output.exists()
    assert source.is_file()


def test_cleanup_never_unlinks_a_replacement_output(source, tmp_path):
    output = tmp_path / "out.mp4"

    def progress(value):
        if value == 100:
            output.unlink()
            output.write_bytes(b"replacement user file")
            raise RuntimeError("UI stopped")

    with pytest.raises(RuntimeError, match="UI stopped"):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, output, progress_callback=progress, watermark_regions=selected()
        )
    assert output.read_bytes() == b"replacement user file"
    assert source.is_file()


def test_cross_volume_publication_copies_then_links_exclusively(source, tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_same_volume", lambda *args: False)
    original_link = core.os.link

    def cross_volume(first, second):
        if Path(second).parent.name.startswith(".watermark-publish-"):
            raise OSError(errno.EXDEV, "different device")
        return original_link(first, second)

    monkeypatch.setattr(core.os, "link", cross_volume)
    output = tmp_path / "output" / "out.mp4"
    receipts = []
    core.WatermarkCleaner(engine=PaintEngine()).run(
        source,
        output,
        watermark_regions=selected(),
        working_directory=tmp_path / "different-drive-work",
        before_publish=lambda partial, dest: receipts.append(SourceIdentity.capture(partial)),
    )
    verify_output(source, output)
    assert output.stat().st_ino == receipts[0].inode
    assert not list(output.parent.glob(".watermark-publish-*"))


def test_drive_without_hard_links_uses_exclusive_native_rename(source, tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "no hard links")

    monkeypatch.setattr(core.os, "link", unsupported)
    output = tmp_path / "out.mp4"
    core.WatermarkCleaner(engine=PaintEngine()).run(source, output, watermark_regions=selected())
    verify_output(source, output)
    assert source.is_file()


def test_native_exclusive_rename_cannot_overwrite_a_racing_file(tmp_path, monkeypatch):
    import ctypes
    import sys

    staged, destination = tmp_path / "staged", tmp_path / "visible"
    staged.write_bytes(b"ours")
    library = ctypes.CDLL(None, use_errno=True)
    name = "renamex_np" if sys.platform == "darwin" else "renameat2"
    native = getattr(library, name)
    native.argtypes = (
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        if name == "renamex_np"
        else [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    )
    native.restype = ctypes.c_int

    class RacingRename:
        def __call__(self, *args):
            destination.write_bytes(b"foreign file")
            return native(*args)

    monkeypatch.setattr(
        core.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(**{name: RacingRename()})
    )
    monkeypatch.setattr(
        core.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "no links")),
    )
    with pytest.raises(FileExistsError):
        core._exclusive_publish(staged, destination)
    assert destination.read_bytes() == b"foreign file"
    assert staged.read_bytes() == b"ours"


def test_symlink_work_directory_is_rejected(source, tmp_path):
    actual = tmp_path / "real"
    actual.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, tmp_path / "out.mp4", watermark_regions=selected(), working_directory=link
        )
    assert source.is_file()
    assert not list(actual.iterdir())


def test_empty_isolated_masks_are_a_failed_job(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        core,
        "build_masks",
        lambda frames, start, regions: np.zeros((len(frames), 80, 120), np.uint8),
    )
    with pytest.raises(ValueError, match="No watermark pixels"):
        core.WatermarkCleaner(engine=PaintEngine()).run(
            source, tmp_path / "out.mp4", watermark_regions=selected()
        )
    assert not (tmp_path / "out.mp4").exists()
    assert source.is_file()
