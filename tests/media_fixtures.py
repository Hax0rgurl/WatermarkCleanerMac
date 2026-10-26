"""Independent tiny H.264/AAC fixtures with explicit variable timestamps."""

from fractions import Fraction
from pathlib import Path
import subprocess

import av
import numpy as np

from watermark_cleaner.video import ffmpeg_path


def create_video(
    path: Path, *, pts=None, durations=None, colors=None, size=(120, 80), audio=False, start=0
):
    pts = pts if pts is not None else [start + index * 3000 for index in range(20)]
    durations = (
        durations if durations is not None else [b - a for a, b in zip(pts, pts[1:])] + [3000]
    )
    colors = colors if colors is not None else [[40 + index % 50] * 3 for index in range(len(pts))]
    time_base = Fraction(1, 90000)
    video = path.with_name(path.stem + ".silent.mp4") if audio else path
    with av.open(str(video), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.codec_context.colorspace = 1
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 1
        stream.options = {"crf": "0", "preset": "ultrafast", "bf": "0"}
        duration_map = dict(zip(pts, durations, strict=True))

        def write_packet(packet):
            stamp = Fraction(packet.pts) * packet.time_base / time_base
            packet.duration = int(duration_map[int(stamp)] * time_base / packet.time_base)
            container.mux(packet)

        for stamp, duration, color in zip(pts, durations, colors, strict=True):
            image = np.full((size[1], size[0], 3), color, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame = frame.reformat(
                format="yuv420p", src_colorspace="ITU709", dst_colorspace="ITU709"
            )
            frame.pts, frame.time_base, frame.duration = stamp, time_base, duration
            for packet in stream.encode(frame):
                write_packet(packet)
        for packet in stream.encode():
            write_packet(packet)
    if audio:
        duration = float((pts[-1] + durations[-1]) * time_base) + 0.15
        subprocess.run(
            [
                ffmpeg_path(),
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(video),
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={duration}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-t",
                str(duration),
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        video.unlink()
    return path
