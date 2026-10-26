"""Sequential video I/O that preserves decoded frame timestamps and source audio."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    total_frames: int
    fps: float
    duration: float
    frame_times: list[float]
    pts: list[int]
    frame_durations: list[int]
    time_base: Fraction
    rate: Fraction
    scene_boundaries: list[int]
    audio_streams: int
    colorspace: int = 1
    color_primaries: int = 1
    color_trc: int = 1
    color_range: int = 1


def ffmpeg_path() -> str:
    configured = os.environ.get("MAC_WATERMARK_FFMPEG")
    if configured and Path(configured).is_file():
        return configured
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=12)
def _probe(path: str, device: int, inode: int, size: int, modified: int, changed: int) -> VideoInfo:
    del device, inode, size, modified, changed
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("The selected file has no video stream.")
        stream = container.streams.video[0]
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError("The video has no valid timestamp time base.")
        tb = Fraction(stream.time_base)
        rate = Fraction(stream.base_rate or stream.average_rate or 30)
        if rate <= 0:
            raise ValueError("The video has an invalid frame rate.")
        pts, durations, cuts = [], [], [0]
        previous = None
        width, height = stream.width, stream.height
        for index, frame in enumerate(container.decode(stream)):
            if frame.width != width or frame.height != height:
                raise ValueError("Videos that change dimensions within a stream are not supported.")
            if frame.pts is None:
                raise ValueError(
                    "The video has missing frame timestamps and cannot be processed safely."
                )
            if frame.time_base is None or frame.time_base <= 0:
                raise ValueError("A video frame has no valid timestamp time base.")
            timestamp = Fraction(frame.pts) * Fraction(frame.time_base) / tb
            if timestamp.denominator != 1:
                raise ValueError("A video frame has an incompatible timestamp time base.")
            value = int(timestamp)
            if pts and value <= pts[-1]:
                raise ValueError("The video has non-increasing frame timestamps.")
            pts.append(value)
            duration = int(frame.duration or max(1, round(1 / rate / tb)))
            if duration < 1:
                raise ValueError("A video frame has an invalid duration.")
            durations.append(duration)
            small = frame.reformat(width=96, height=170, format="rgb24").to_ndarray()
            if previous is not None:
                difference = np.abs(small.astype(np.int16) - previous.astype(np.int16)).mean()
                if difference > 30:
                    cuts.append(index)
            previous = small
        if not pts:
            raise ValueError("No readable video frames were found.")
        for index in range(len(pts) - 1):
            durations[index] = pts[index + 1] - pts[index]
        end = float((pts[-1] + durations[-1]) * tb)
        duration = max(end, float(container.duration or 0) / av.time_base)
        codec = stream.codec_context
        return VideoInfo(
            width,
            height,
            len(pts),
            len(pts) / (end - float(pts[0] * tb)),
            duration,
            [float(p * tb) for p in pts],
            pts,
            durations,
            tb,
            rate,
            cuts + [len(pts)],
            len(container.streams.audio),
            int(codec.colorspace or 1),
            int(codec.color_primaries or 1),
            int(codec.color_trc or 1),
            int(codec.color_range or 1),
        )


def probe_video(path: Path | str) -> VideoInfo:
    path = Path(path).expanduser()
    stat = path.stat()
    return _probe(
        str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def iter_frames(path: Path | str) -> Iterator[np.ndarray]:
    """Decode in order. Seeking by average frame rate breaks variable-rate clips."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


class VideoWriter:
    """Encode once at source dimensions with one explicit timestamp per frame."""

    def __init__(self, path: Path, info: VideoInfo):
        self.info = info
        self.count = 0
        self._durations_by_pts = dict(zip(info.pts, info.frame_durations, strict=True))
        self.container = av.open(str(path), "w", format="mp4")
        try:
            self.stream = self.container.add_stream("libx264", rate=info.rate)
            self.stream.width, self.stream.height = info.width, info.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = info.time_base
            self.stream.codec_context.time_base = info.time_base
            # A one-to-one presentation/decode order keeps explicit VFR sample
            # durations, including a held last frame, intact through MP4 remuxing.
            self.stream.options = {"crf": "16", "preset": "medium", "bf": "0"}
            codec = self.stream.codec_context
            codec.colorspace = info.colorspace
            codec.color_primaries = info.color_primaries
            codec.color_trc = info.color_trc
            codec.color_range = info.color_range
        except BaseException:
            self.container.close()
            raise
        self.closed = False

    def write(self, image: np.ndarray) -> None:
        if self.count >= self.info.total_frames:
            raise ValueError("The input gained frames during processing.")
        if image.shape != (self.info.height, self.info.width, 3) or image.dtype != np.uint8:
            raise ValueError("A restored video frame has invalid dimensions or pixel type.")
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        # BT.709/BT.601 metadata must describe the conversion actually used.
        matrix = {1: "ITU709", 4: "FCC", 7: "SMPTE240M", 9: "BT2020"}.get(
            self.info.colorspace, "ITU601"
        )
        frame = frame.reformat(
            format="yuv420p",
            src_colorspace=matrix,
            dst_colorspace=matrix,
            src_color_range="JPEG",
            dst_color_range="JPEG" if self.info.color_range == 2 else "MPEG",
        )
        frame.pts = self.info.pts[self.count]
        frame.time_base = self.info.time_base
        frame.duration = self.info.frame_durations[self.count]
        for packet in self.stream.encode(frame):
            self._mux(packet)
        self.count += 1

    def _mux(self, packet) -> None:
        if packet.pts is None or packet.time_base is None:
            raise ValueError("The encoder did not retain a video frame timestamp.")
        stamp = Fraction(packet.pts) * packet.time_base / self.info.time_base
        if stamp.denominator != 1 or int(stamp) not in self._durations_by_pts:
            raise ValueError("The encoder changed a video frame timestamp.")
        duration = self._durations_by_pts[int(stamp)] * self.info.time_base / packet.time_base
        if duration.denominator != 1:
            raise ValueError("The encoder changed a video frame duration.")
        packet.duration = int(duration)
        self.container.mux(packet)

    def close(self, *, complete: bool = True) -> None:
        if self.closed:
            return
        try:
            if complete and self.count != self.info.total_frames:
                raise ValueError(
                    f"Only {self.count} of {self.info.total_frames} video frames were processed."
                )
            if complete:
                for packet in self.stream.encode():
                    self._mux(packet)
        finally:
            try:
                self.container.close()
            finally:
                self.closed = True


def mux_audio(source: Path, video: Path, output: Path, info: VideoInfo) -> None:
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-copyts",
        "-i",
        str(video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "disabled",
        "-movflags",
        "+faststart",
    ]
    if info.time_base.numerator == 1:
        command += ["-video_track_timescale", str(info.time_base.denominator)]
    command += [str(output)]
    subprocess.run(command, capture_output=True, text=True, timeout=3600, check=True)


@lru_cache(maxsize=12)
def _audio_records(path: str, device: int, inode: int, size: int, modified: int, changed: int):
    del device, inode, size, modified, changed
    records = []
    with av.open(path) as container:
        for index, stream in enumerate(container.streams.audio):
            digest = hashlib.sha256()
            timestamps = []
            with av.open(path) as reader:
                for packet in reader.demux(reader.streams.audio[index]):
                    if packet.size:
                        digest.update(bytes(packet))
                        timestamps.append(
                            tuple(
                                None if stamp is None else Fraction(stamp) * packet.time_base
                                for stamp in (packet.pts, packet.dts)
                            )
                        )
            start = (
                None
                if stream.start_time is None
                else Fraction(stream.start_time) * stream.time_base
            )
            duration = (
                None if stream.duration is None else Fraction(stream.duration) * stream.time_base
            )
            codec = stream.codec_context
            timing = (
                codec.name,
                codec.sample_rate,
                codec.layout.name,
                start,
                duration,
                tuple(timestamps),
            )
            records.append((digest.hexdigest(), timing))
    return tuple(records)


def _audio_properties(path: Path):
    path = Path(path).expanduser()
    stat = path.stat()
    return _audio_records(
        str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def audio_signatures(path: Path) -> list[str]:
    """Hash original compressed audio packets without changing audio codecs."""
    return [record[0] for record in _audio_properties(path)]


def audio_timing(path: Path) -> list[tuple]:
    """Capture effective stream bounds and exact packet timestamps for sync checks."""
    return [record[1] for record in _audio_properties(path)]


def verify_output(source: Path, output: Path, expected: VideoInfo | None = None) -> None:
    expected = expected or probe_video(source)
    actual = probe_video(output)
    if (actual.width, actual.height, actual.total_frames) != (
        expected.width,
        expected.height,
        expected.total_frames,
    ):
        raise ValueError("The cleaned video has different dimensions or is missing frames.")
    before = [p * expected.time_base for p in expected.pts]
    after = [p * actual.time_base for p in actual.pts]
    if before != after:
        raise ValueError("The cleaned video changed the original frame timing.")
    if (
        expected.frame_durations[-1] * expected.time_base
        != actual.frame_durations[-1] * actual.time_base
    ):
        raise ValueError("The cleaned video changed the final frame duration.")
    if audio_signatures(source) != audio_signatures(output):
        raise ValueError("The cleaned video did not preserve the original audio.")
    if audio_timing(source) != audio_timing(output):
        raise ValueError("The cleaned video changed the original audio timing.")
    subprocess.run(
        [
            ffmpeg_path(),
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=True,
    )
