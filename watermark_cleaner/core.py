"""Bounded, shot-aware temporal restoration with verified publication."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import replace
from datetime import datetime
import errno
import hashlib
import os
import shutil
import sys
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from .engines import EngineName, get_engine
from .completion import SourceIdentity, SourceIdentityError, validate_source_identity
from .masks import build_masks, context_crop
from .scanner import scan_video
from .timed_regions import normalize_regions, compile_regions, region_boxes
from .templates import validate_style
from .video import iter_frames, probe_video, VideoWriter, mux_audio, verify_output


SOURCE_BUFFER_BUDGET_BYTES = 400_000_000


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_volume(path: Path, directory: Path) -> bool:
    return path.stat().st_dev == directory.stat().st_dev


@contextmanager
def _publication_stage(complete: Path, directory: Path):
    """Prepare the verified file on the output volume before exposing its name."""
    if _same_volume(complete, directory):
        yield complete
        return
    with tempfile.TemporaryDirectory(prefix=".watermark-publish-", dir=directory) as temporary:
        staged = Path(temporary) / "complete.mp4"
        try:
            os.link(complete, staged)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise
            with complete.open("rb") as source, staged.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if _digest(staged) != _digest(complete):
                raise RuntimeError(
                    "The cleaned file could not be copied safely to the output drive."
                )
        yield staged


def _exclusive_publish(staged: Path, destination: Path) -> None:
    """Publish atomically without replacing a file, including on external drives."""
    try:
        os.link(staged, destination)
        return
    except OSError as error:
        if error.errno == errno.EEXIST or destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"The output path appeared during processing: {destination}"
            ) from error
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise
    # FAT/exFAT do not provide hard links. Use the platform's exclusive rename,
    # never ordinary POSIX rename, which could overwrite a concurrently made file.
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        operation = library.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(os.fsencode(staged), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        operation = library.renameat2
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-100, os.fsencode(staged), -100, os.fsencode(destination), 1)
    elif os.name == "nt":
        os.rename(staged, destination)  # Windows rename already refuses an existing target.
        return
    else:
        raise RuntimeError(
            "This output drive cannot publish files safely. Choose another output folder."
        )
    if result:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"The output path appeared during processing: {destination}")
        raise OSError(code, os.strerror(code), str(destination))


def _remove_owned_output(identity: SourceIdentity) -> None:
    """A failure may remove our unchanged output, never a replacement user file."""
    try:
        if SourceIdentity.capture(identity.source_path) == identity:
            identity.source_path.unlink(missing_ok=True)
    except (OSError, SourceIdentityError):
        pass


def unique_mp4_path(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = directory / f"{stem}-cleaned-{stamp}.mp4"
    counter = 2
    while output.exists() or output.is_symlink():
        output = directory / f"{stem}-cleaned-{stamp}-{counter}.mp4"
        counter += 1
    return output


def _tracks(regions: list[dict]) -> list[list[dict]]:
    tracks = []
    for region in regions:
        box = region.get("anchor_bbox", region["bbox"])
        found = None
        for track in tracks:
            other = track[0].get("anchor_bbox", track[0]["bbox"])
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1])
            )
            small = min(
                (box[2] - box[0]) * (box[3] - box[1]), (other[2] - other[0]) * (other[3] - other[1])
            )
            if small and intersection / small >= 0.7:
                found = track
                break
        if found is None:
            tracks.append([region])
        else:
            found.append(region)
    return tracks


class WatermarkCleaner:
    def __init__(self, engine_id: str = "e2fgvi", *, engine=None):
        self.engine_id = EngineName(engine_id).value
        self._engine = engine

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
        working_directory: Path | None = None,
        before_publish: Callable | None = None,
        watermark_regions: list | None = None,
        watermark_text: str = "",
        watermark_style: str = "auto",
    ) -> None:
        del quiet
        watermark_style = validate_style(watermark_style)
        source, destination = (
            Path(input_video_path).expanduser(),
            Path(output_video_path).expanduser(),
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == destination.resolve():
            raise ValueError("The source and cleaned output must be different files.")
        if destination.suffix.lower() != ".mp4":
            raise ValueError("The cleaned output must use the .mp4 extension.")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        source_identity = SourceIdentity.capture(source)
        info = probe_video(source)
        if info.width % 2 or info.height % 2:
            raise ValueError("MP4 restoration currently requires even video dimensions.")
        progress = progress_callback or (lambda value: None)
        progress(0)
        if watermark_regions is None:
            scan = scan_video(
                source,
                watermark_text,
                progress=lambda value, text: progress(int(value * 15)),
                watermark_style=watermark_style,
            )
            watermark_regions = scan["regions"]
        if not watermark_regions:
            raise ValueError(
                "No matching watermark was confirmed. Enter its name or use the visual player to mark it."
            )
        valid = normalize_regions(watermark_regions, info.total_frames)
        regions = [
            {**original, **normalized} if isinstance(original, dict) else normalized
            for original, normalized in zip(watermark_regions, valid, strict=True)
        ]
        compile_regions(regions, info.width, info.height, info.total_frames)
        for region in regions:
            if "mask_parts" in region:
                region_boxes(region["mask_parts"], info.width, info.height)
            for key in ("anchor_bbox", "icon_search"):
                if key in region:
                    region_boxes([region[key]], info.width, info.height)
                    region[key] = list(map(float, region[key]))
        tracks = _tracks(regions)
        engine = self._engine or get_engine(self.engine_id)
        parent = Path(working_directory) if working_directory else destination.parent
        if parent.is_symlink():
            raise ValueError("The job working directory must be a real directory.")
        parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Bounded source buffers: at most ~400MB, usually much smaller context crops.
        capacity = max(1, min(80, SOURCE_BUFFER_BUDGET_BYTES // (info.width * info.height * 3)))
        overlap = min(12, capacity // 4)
        published_identity = None
        with tempfile.TemporaryDirectory(prefix="temporal-", dir=parent) as temporary:
            work = Path(temporary)
            writer = VideoWriter(work / "video.mp4", info)
            decoder = iter_frames(source)
            written = 0
            cleaned_pixels = 0
            previous_tail = None
            previous_mask = None
            try:
                for shot_start, shot_end in zip(
                    info.scene_boundaries[:-1], info.scene_boundaries[1:], strict=True
                ):
                    carry = []
                    cursor = shot_start
                    previous_tail, previous_mask = None, None
                    while cursor < shot_end:
                        length = min(capacity, shot_end - cursor)
                        try:
                            frames = carry + [next(decoder) for _ in range(length - len(carry))]
                        except StopIteration as error:
                            raise ValueError(
                                "The input video ended before all of its frames were processed."
                            ) from error
                        active_tracks = [
                            track
                            for track in tracks
                            if any(
                                r["start_frame"] < cursor + length and r["end_frame"] > cursor
                                for r in track
                            )
                        ]
                        restored = [frame.copy() for frame in frames]
                        union_mask = np.zeros((length, info.height, info.width), dtype=bool)
                        commit = length if cursor + length == shot_end else length - overlap
                        for track_index, track in enumerate(active_tracks):
                            masks = build_masks(frames, cursor, track)
                            if not masks.any():
                                continue
                            union_mask |= masks > 0
                            left, top, right, bottom = context_crop(masks, info.width, info.height)
                            inputs = np.stack([frame[top:bottom, left:right] for frame in frames])
                            cropped_masks = masks[:, top:bottom, left:right]

                            def model_progress(value, text):
                                fraction = (track_index + float(value)) / max(1, len(active_tracks))
                                progress(
                                    min(
                                        89,
                                        15
                                        + int(
                                            74 * (written + commit * fraction) / info.total_frames
                                        ),
                                    )
                                )

                            prediction = engine.restore(
                                inputs, cropped_masks, progress=model_progress
                            )
                            if prediction.shape != inputs.shape or prediction.dtype != np.uint8:
                                raise ValueError(
                                    "The restoration model returned an invalid frame sequence."
                                )
                            for index in range(length):
                                selected = cropped_masks[index] > 0
                                restored[index][top:bottom, left:right][selected] = prediction[
                                    index
                                ][selected]
                        if previous_tail is not None:
                            for index in range(min(len(previous_tail), length)):
                                selected = union_mask[index] & previous_mask[index]
                                weight = (index + 1) / (len(previous_tail) + 1)
                                restored[index][selected] = np.rint(
                                    previous_tail[index][selected] * (1 - weight)
                                    + restored[index][selected] * weight
                                ).astype(np.uint8)
                        for index in range(length):
                            if not np.array_equal(
                                restored[index][~union_mask[index]],
                                frames[index][~union_mask[index]],
                            ):
                                raise RuntimeError(
                                    "Restoration altered pixels outside the watermark mask."
                                )
                        cleaned_pixels += int(np.count_nonzero(union_mask[:commit]))
                        for frame in restored[:commit]:
                            writer.write(frame)
                        written += commit
                        progress(15 + int(74 * written / info.total_frames))
                        previous_tail = restored[commit:]
                        previous_mask = union_mask[commit:]
                        carry = frames[commit:]
                        cursor += commit
                try:
                    next(decoder)
                except StopIteration:
                    pass
                else:
                    raise ValueError("The source gained frames during processing.")
                if not cleaned_pixels:
                    raise ValueError(
                        "No watermark pixels could be isolated. The original video was kept."
                    )
                writer.close()
                validate_source_identity(source_identity)
                progress(91)
                mux_audio(source, work / "video.mp4", work / "complete.mp4", info)
                validate_source_identity(source_identity)
                progress(95)
                verify_output(source, work / "complete.mp4", info)
                validate_source_identity(source_identity)
                with _publication_stage(work / "complete.mp4", destination.parent) as staged:
                    staged_identity = SourceIdentity.capture(staged)
                    if before_publish:
                        before_publish(staged, destination)
                    validate_source_identity(staged_identity)
                    validate_source_identity(source_identity)
                    _exclusive_publish(staged, destination)
                    published_identity = replace(staged_identity, path=str(destination.absolute()))
                    validate_source_identity(published_identity)
                    progress(100)
            except BaseException:
                if not writer.closed:
                    try:
                        writer.close(complete=False)
                    except Exception:
                        pass
                if published_identity is not None:
                    _remove_owned_output(published_identity)
                raise
            finally:
                decoder.close()
