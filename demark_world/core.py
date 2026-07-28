"""Universal video watermark removal pipeline.

Derived from DeMark-World and restructured for reliable desktop operation. The
entire processing phase is guarded so any exception terminates FFmpeg and
removes incomplete output.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Callable

import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm

from demark_world.configs import (
    DEFAULT_DETECT_BATCH_SIZE,
    ENABLE_E2FGVI_HQ_TORCH_COMPILE,
)
from demark_world.schemas import CleanerType
from demark_world.utils.devices_utils import is_bf16_supported
from demark_world.utils.imputation_utils import (
    find_2d_data_bkps,
    find_idxs_interval,
    get_interval_average_bbox,
    refine_bkps_by_chunk_size,
)
from demark_world.utils.video_utils import VideoLoader, merge_frames_with_overlap
from demark_world.watermark_cleaner import WaterMarkCleaner
from demark_world.watermark_detector import DeMarkWorldDetector

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]


def unique_mp4_path(directory: Path, stem: str) -> Path:
    """Return a collision-safe MP4 path without touching existing files."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    candidate = directory / f"{stem}-cleaned-{timestamp}.mp4"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-cleaned-{timestamp}-{counter}.mp4"
        counter += 1
    return candidate


class DeMarkWorld:
    def __init__(
        self,
        cleaner_type: CleanerType = CleanerType.LAMA,
        enable_torch_compile: bool = ENABLE_E2FGVI_HQ_TORCH_COMPILE,
        detect_batch_size: int = DEFAULT_DETECT_BATCH_SIZE,
        use_bf16: bool = is_bf16_supported(),
        detector: Any | None = None,
        cleaner: Any | None = None,
    ):
        if detect_batch_size < 1:
            raise ValueError("detect_batch_size must be at least 1")
        self.detector = detector if detector is not None else DeMarkWorldDetector()
        self.cleaner = (
            cleaner
            if cleaner is not None
            else WaterMarkCleaner(
                cleaner_type,
                enable_torch_compile,
                use_bf16=use_bf16,
            )
        )
        self.cleaner_type = cleaner_type
        self.detect_batch_size = detect_batch_size

    @staticmethod
    def _abort_encoding(
        process_out: Any,
        temp_output_path: Path,
    ) -> None:
        """Stop an in-flight encoder and remove every partial output."""
        try:
            if process_out.stdin and not process_out.stdin.closed:
                process_out.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            if process_out.poll() is None:
                process_out.terminate()
                try:
                    process_out.wait(timeout=5)
                except Exception:
                    process_out.kill()
                    process_out.wait(timeout=5)
        finally:
            temp_output_path.unlink(missing_ok=True)

    def run_batch(
        self,
        input_video_dir_path: Path,
        output_video_dir_path: Path | None = None,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
    ) -> None:
        input_video_dir_path = Path(input_video_dir_path).expanduser()
        if not input_video_dir_path.is_dir():
            raise NotADirectoryError(input_video_dir_path)
        if output_video_dir_path is None:
            output_video_dir_path = input_video_dir_path.parent / "watermark_removed"
        output_video_dir_path = Path(output_video_dir_path).expanduser()
        output_video_dir_path.mkdir(parents=True, exist_ok=True)

        inputs = sorted(
            path
            for path in input_video_dir_path.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        logger.info(f"Found {len(inputs)} video(s) to process")
        for index, input_path in enumerate(
            tqdm(inputs, desc="Processing videos", disable=quiet)
        ):
            # MP4 is the reliable output container for the H.264/AAC encoder.
            output_path = unique_mp4_path(output_video_dir_path, input_path.stem)

            def batch_progress(percent: int) -> None:
                if progress_callback and inputs:
                    overall = int(((index + percent / 100) / len(inputs)) * 100)
                    progress_callback(min(overall, 100))

            self.run(
                input_path,
                output_path,
                progress_callback=batch_progress if progress_callback else None,
                quiet=quiet,
            )

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
        working_directory: Path | None = None,
        before_publish: Callable[[Path, Path], None] | None = None,
    ) -> None:
        input_video_path = Path(input_video_path).expanduser()
        output_video_path = Path(output_video_path).expanduser()
        if not input_video_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_video_path}")
        if input_video_path.resolve() == output_video_path.resolve():
            raise ValueError("Input and output paths must be different")
        if output_video_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_video_path}"
            )

        loader = VideoLoader(input_video_path)
        width, height, fps = loader.width, loader.height, loader.fps
        total_frames = loader.total_frames
        if total_frames < 1 or width < 1 or height < 1:
            raise ValueError(f"No readable video frames found in {input_video_path}")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        if working_directory is None:
            artifact_directory = output_video_path.parent
            temp_name = f".{output_video_path.stem}.partial{output_video_path.suffix}"
        else:
            artifact_directory = Path(working_directory).expanduser()
            artifact_directory.mkdir(parents=True, exist_ok=True)
            if not os.path.isdir(artifact_directory) or os.path.islink(artifact_directory):
                raise ValueError("Job working directory must be a real directory")
            temp_name = "video.partial.mp4"
        temp_output_path = artifact_directory / temp_name
        temp_output_path.unlink(missing_ok=True)

        output_options: dict[str, str] = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",
        }
        if loader.original_bitrate:
            output_options["video_bitrate"] = str(
                int(int(loader.original_bitrate) * 1.2)
            )
        else:
            output_options["crf"] = "18"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        try:
            bboxes, breakpoints = self._detect_bboxes(
                input_video_path,
                total_frames,
                progress_callback,
                quiet,
            )
            if self.cleaner_type == CleanerType.LAMA:
                self._clean_lama(
                    input_video_path,
                    bboxes,
                    width,
                    height,
                    total_frames,
                    process_out,
                    progress_callback,
                    quiet,
                )
            elif self.cleaner_type == CleanerType.E2FGVI_HQ:
                self._clean_temporal(
                    input_video_path,
                    bboxes,
                    breakpoints,
                    width,
                    height,
                    total_frames,
                    process_out,
                    progress_callback,
                    quiet,
                )
            else:
                raise ValueError(f"Unsupported cleaner type: {self.cleaner_type}")

            process_out.stdin.close()
            status = process_out.wait()
            if status != 0 or not temp_output_path.is_file():
                raise RuntimeError(
                    "FFmpeg could not encode the cleaned video. Check that the "
                    "input is supported and review the app log."
                )
            if progress_callback:
                progress_callback(95)
            if working_directory is None:
                self.merge_audio_track(
                    input_video_path,
                    temp_output_path,
                    output_video_path,
                    before_publish=before_publish,
                )
            else:
                self.merge_audio_track(
                    input_video_path,
                    temp_output_path,
                    output_video_path,
                    working_directory=artifact_directory,
                    before_publish=before_publish,
                )
            if progress_callback:
                progress_callback(100)
        except BaseException:
            self._abort_encoding(
                process_out,
                temp_output_path,
            )
            raise

    def _detect_bboxes(
        self,
        input_video_path: Path,
        total_frames: int,
        progress_callback: Callable[[int], None] | None,
        quiet: bool,
    ) -> tuple[list[tuple[int, int, int, int] | None], list[int]]:
        bboxes: list[tuple[int, int, int, int] | None] = [None] * total_frames
        centers: list[tuple[int, int] | None] = [None] * total_frames
        loader = VideoLoader(input_video_path)
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []

        def flush() -> None:
            if not batch_frames:
                return
            results = self.detector.detect_batch(
                batch_frames,
                batch_size=self.detect_batch_size,
            )
            if len(results) != len(batch_frames):
                raise RuntimeError(
                    "Detector returned a different number of results than frames."
                )
            for frame_index, result in zip(batch_indices, results):
                if result["detected"] and result["bbox"] is not None:
                    bbox = tuple(int(value) for value in result["bbox"])
                    bboxes[frame_index] = bbox
                    x1, y1, x2, y2 = bbox
                    centers[frame_index] = ((x1 + x2) // 2, (y1 + y2) // 2)
                if progress_callback and frame_index % 10 == 0:
                    progress_callback(10 + int((frame_index / total_frames) * 40))
            batch_frames.clear()
            batch_indices.clear()

        for index, frame in enumerate(
            tqdm(loader, total=total_frames, desc="Detect watermarks", disable=quiet)
        ):
            if index >= total_frames:
                break
            batch_frames.append(frame)
            batch_indices.append(index)
            if len(batch_frames) >= self.detect_batch_size:
                flush()
        flush()

        valid_indices = [index for index, bbox in enumerate(bboxes) if bbox is not None]
        if not valid_indices:
            return bboxes, [0, total_frames]

        missed = [index for index, bbox in enumerate(bboxes) if bbox is None]
        breakpoints = [0, total_frames]
        if missed:
            try:
                detected_breakpoints = find_2d_data_bkps(centers)
                breakpoints = sorted(
                    {
                        0,
                        total_frames,
                        *(
                            int(point)
                            for point in detected_breakpoints
                            if 0 < int(point) < total_frames
                        ),
                    }
                )
                interval_bboxes = get_interval_average_bbox(bboxes, breakpoints)
                intervals = find_idxs_interval(missed, breakpoints)
                for frame_index, interval_index in zip(missed, intervals):
                    if (
                        interval_index < len(interval_bboxes)
                        and interval_bboxes[interval_index] is not None
                    ):
                        bboxes[frame_index] = tuple(
                            int(value) for value in interval_bboxes[interval_index]
                        )
            except Exception as exc:
                logger.warning(f"Interval bbox imputation failed; using nearest: {exc}")

            # A nearest valid box is a deterministic fallback for any remaining
            # gaps, including beginning/end runs.
            for frame_index in missed:
                if bboxes[frame_index] is None:
                    nearest = min(valid_indices, key=lambda value: abs(value - frame_index))
                    bboxes[frame_index] = bboxes[nearest]
        return bboxes, breakpoints

    def _clean_lama(
        self,
        input_video_path: Path,
        bboxes: list[tuple[int, int, int, int] | None],
        width: int,
        height: int,
        total_frames: int,
        process_out: Any,
        progress_callback: Callable[[int], None] | None,
        quiet: bool,
    ) -> None:
        for index, frame in enumerate(
            tqdm(
                VideoLoader(input_video_path),
                total=total_frames,
                desc="Remove watermarks",
                disable=quiet,
            )
        ):
            if index >= total_frames:
                break
            bbox = bboxes[index]
            cleaned = frame
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 255
                cleaned = self.cleaner.clean(frame, mask)
            process_out.stdin.write(cleaned.astype(np.uint8).tobytes())
            if progress_callback and index % 10 == 0:
                progress_callback(50 + int((index / total_frames) * 45))

    def _clean_temporal(
        self,
        input_video_path: Path,
        bboxes: list[tuple[int, int, int, int] | None],
        breakpoints: list[int],
        width: int,
        height: int,
        total_frames: int,
        process_out: Any,
        progress_callback: Callable[[int], None] | None,
        quiet: bool,
    ) -> None:
        loader = VideoLoader(input_video_path)
        refined = refine_bkps_by_chunk_size(breakpoints, self.cleaner.chunk_size)
        result_frames = None
        written = 0
        for segment_index in range(len(refined) - 1):
            core_start, core_end = refined[segment_index : segment_index + 2]
            segment_length = core_end - core_start
            overlap = max(
                1,
                int(float(self.cleaner.config.overlap_ratio) * segment_length),
            )
            start = max(0, core_start - (overlap if segment_index else 0))
            end = min(
                total_frames,
                core_end + (overlap if segment_index < len(refined) - 2 else 0),
            )
            frames = np.asarray(loader.get_slice(start, end))
            frames_rgb = frames[:, :, :, ::-1].copy()
            masks = np.zeros((len(frames), height, width), dtype=np.uint8)
            for absolute_index in range(start, end):
                bbox = bboxes[absolute_index]
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    masks[absolute_index - start][
                        max(0, y1) : min(height, y2),
                        max(0, x1) : min(width, x2),
                    ] = 255
            cleaned = self.cleaner.clean(frames_rgb, masks)
            result_frames = merge_frames_with_overlap(
                result_frames,
                cleaned,
                start,
                overlap,
                is_first_chunk=segment_index == 0,
            )
            for absolute_index in range(core_start, core_end):
                frame = result_frames[absolute_index]
                if frame is None:
                    raise RuntimeError("Temporal cleaner returned a missing frame")
                process_out.stdin.write(
                    frame[:, :, ::-1].astype(np.uint8).tobytes()
                )
                written += 1
                if progress_callback and written % 10 == 0:
                    progress_callback(50 + int((written / total_frames) * 45))

    def merge_audio_track(
        self,
        input_video_path: Path,
        temp_output_path: Path,
        output_video_path: Path,
        working_directory: Path | None = None,
        before_publish: Callable[[Path, Path], None] | None = None,
    ) -> None:
        logger.info("Merging audio track…")
        if output_video_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_video_path}"
            )
        if working_directory is None:
            merged_partial = output_video_path.parent / (
                f".{output_video_path.stem}.audio-partial{output_video_path.suffix}"
            )
        else:
            merged_partial = Path(working_directory) / "audio.partial.mp4"
        merged_partial.unlink(missing_ok=True)
        try:
            video = ffmpeg.input(str(temp_output_path))
            audio = ffmpeg.input(str(input_video_path)).audio
            (
                ffmpeg.output(
                    video,
                    audio,
                    str(merged_partial),
                    vcodec="copy",
                    acodec="aac",
                )
                .overwrite_output()
                .global_args("-loglevel", "error")
                .run(capture_stdout=True, capture_stderr=True)
            )
            self._publish_no_clobber(
                merged_partial,
                output_video_path,
                before_publish=before_publish,
            )
            temp_output_path.unlink(missing_ok=True)
            logger.info(f"Saved cleaned video with audio: {output_video_path}")
        except ffmpeg.Error as exc:
            details = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
            logger.info(f"Keeping silent cleaned video; audio merge detail: {details}")
            self._publish_no_clobber(
                temp_output_path,
                output_video_path,
                before_publish=before_publish,
            )
        finally:
            merged_partial.unlink(missing_ok=True)

    @staticmethod
    def _publish_no_clobber(
        source: Path,
        destination: Path,
        *,
        before_publish: Callable[[Path, Path], None] | None = None,
    ) -> None:
        """Atomically publish a same-volume file without replacing anything."""
        if before_publish is not None:
            before_publish(source, destination)
        try:
            os.link(source, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {destination}"
            ) from exc
        source.unlink()
