"""Video loading helpers.

Changed from DeMark-World to use OpenCV for metadata and frame iteration. This
removes an ffprobe dependency from the self-contained macOS runtime; FFmpeg is
still used by the core encoder to preserve output quality and audio.
"""

import math
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np


def merge_frames_with_overlap(
    result_frames: Optional[List[Optional[np.ndarray]]],
    chunk_frames: List[np.ndarray],
    start_idx: int,
    overlap_size: int,
    is_first_chunk: bool = False,
) -> List[Optional[np.ndarray]]:
    chunk_size = len(chunk_frames)
    if result_frames is None or is_first_chunk:
        if result_frames is None:
            result_frames = [None] * (start_idx + chunk_size)
        for index in range(chunk_size):
            result_frames[start_idx + index] = chunk_frames[index]
        return result_frames

    required_size = start_idx + chunk_size
    if len(result_frames) < required_size:
        result_frames.extend([None] * (required_size - len(result_frames)))

    overlap_end = min(max(0, overlap_size), chunk_size)
    for index in range(overlap_end):
        result_index = start_idx + index
        if result_frames[result_index] is not None and chunk_frames[index] is not None:
            alpha = index / overlap_end if overlap_end else 1.0
            result_frames[result_index] = (
                result_frames[result_index].astype(np.float32) * (1 - alpha)
                + chunk_frames[index].astype(np.float32) * alpha
            ).astype(np.uint8)
        elif chunk_frames[index] is not None:
            result_frames[result_index] = chunk_frames[index]

    for index in range(overlap_end, chunk_size):
        result_frames[start_idx + index] = chunk_frames[index]
    return result_frames


class VideoLoader:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self._read_video_info()

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"OpenCV could not open video: {self.video_path}")
        return capture

    def _read_video_info(self) -> None:
        capture = self._open()
        try:
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Input video has invalid frame rate")
            self.total_frames = self._count_decoded_frames(capture)
            bitrate = int(capture.get(cv2.CAP_PROP_BITRATE))
            self.original_bitrate = str(bitrate * 1000) if bitrate > 0 else None
        finally:
            capture.release()

    @staticmethod
    def _count_decoded_frames(capture: cv2.VideoCapture) -> int:
        """Count successful reads rather than trusting container metadata."""
        count = 0
        while capture.read()[0]:
            count += 1
        return count

    def __len__(self) -> int:
        return self.total_frames

    def get_slice(self, start: int, end: int) -> List[np.ndarray]:
        if end <= start:
            return []
        capture = self._open()
        frames: List[np.ndarray] = []
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(end - start):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def __iter__(self) -> Iterator[np.ndarray]:
        capture = self._open()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame
        finally:
            capture.release()
