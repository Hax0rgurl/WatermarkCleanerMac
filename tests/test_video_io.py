"""Run real encoding, decoding, timestamp, audio and format checks."""

from fractions import Fraction
from pathlib import Path
import subprocess

import av
import numpy as np
import pytest

from tests.media_fixtures import create_video
from watermark_cleaner import video


def round_trip(source, target):
    info = video.probe_video(source)
    writer = video.VideoWriter(target, info)
    for frame in video.iter_frames(source):
        writer.write(frame)
    writer.close()
    return info


def test_variable_frame_timing_and_all_audio_packets_survive(tmp_path):
    pts = [0, 3000, 6000, 12000, 15000, 21000, 24000, 30000]
    source = create_video(tmp_path / "vfr.mp4", pts=pts, audio=True)
    info = round_trip(source, tmp_path / "video.mp4")
    output = tmp_path / "complete.mp4"
    video.mux_audio(source, tmp_path / "video.mp4", output, info)
    video.verify_output(source, output, info)
    actual = video.probe_video(output)
    assert info.total_frames == actual.total_frames == len(pts)
    assert [p * info.time_base for p in info.pts] == [Fraction(p, 90000) for p in pts]
    assert [p * actual.time_base for p in actual.pts] == [Fraction(p, 90000) for p in pts]
    assert video.audio_signatures(source) == video.audio_signatures(output)
    assert actual.audio_streams == 1
    assert actual.duration > float(actual.pts[-1] * actual.time_base)


def test_nonzero_video_start_is_preserved(tmp_path):
    source = create_video(tmp_path / "offset.mp4", pts=[9000, 12000, 18000, 21000])
    info = round_trip(source, tmp_path / "copy.mp4")
    video.verify_output(source, tmp_path / "copy.mp4", info)
    assert info.frame_times[0] == 0.1


@pytest.mark.parametrize("with_audio", [False, True])
def test_public_pipeline_preserves_fractional_nonzero_start_and_audio(tmp_path, with_audio):
    from watermark_cleaner.core import WatermarkCleaner

    pts = [441441, 444444, 447447, 450450]
    source = create_video(tmp_path / "offset.mp4", pts=pts)
    if with_audio:
        tone = create_video(tmp_path / "tone.mp4", audio=True)
        combined = tmp_path / "offset-with-audio.mp4"
        with (
            av.open(str(source)) as pictures,
            av.open(str(tone)) as sound,
            av.open(str(combined), "w", options={"movie_timescale": "4410000"}) as destination,
        ):
            picture_stream = destination.add_stream_from_template(pictures.streams.video[0])
            audio_stream = destination.add_stream_from_template(sound.streams.audio[0])
            for packet in pictures.demux(video=0):
                if packet.size:
                    packet.stream = picture_stream
                    destination.mux(packet)
            for packet in sound.demux(audio=0):
                if packet.size:
                    # A deliberate 4.9-second audio offset also tests AAC trim
                    # metadata and effective stream bounds after remuxing.
                    offset = int(Fraction(49, 10) / packet.time_base)
                    packet.pts += offset
                    packet.dts += offset
                    packet.stream = audio_stream
                    destination.mux(packet)
        source = combined
    original = source.read_bytes()
    before = video.probe_video(source)
    assert [stamp * before.time_base for stamp in before.pts] == [
        Fraction(stamp, 90000) for stamp in pts
    ]

    class IdentityEngine:
        def restore(self, frames, masks, progress=None):
            return frames.copy()

    output = tmp_path / "cleaned.mp4"
    WatermarkCleaner(engine=IdentityEngine()).run(
        source,
        output,
        watermark_regions=[{"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 0, "end_frame": 4}],
    )
    video.verify_output(source, output, before)
    after = video.probe_video(output)
    assert [stamp * after.time_base for stamp in after.pts] == [
        Fraction(stamp, 90000) for stamp in pts
    ]
    assert video.audio_signatures(source) == video.audio_signatures(output)
    assert video.audio_timing(source) == video.audio_timing(output)
    assert after.audio_streams == int(with_audio)
    assert source.read_bytes() == original


def test_long_final_frame_duration_is_preserved(tmp_path):
    source = create_video(tmp_path / "held.mp4", pts=[0, 3000, 6000], durations=[3000, 3000, 18000])
    info = round_trip(source, tmp_path / "copy.mp4")
    actual = video.probe_video(tmp_path / "copy.mp4")
    assert info.frame_durations[-1] * info.time_base == Fraction(1, 5)
    assert actual.frame_durations[-1] * actual.time_base == Fraction(1, 5)


def test_equal_luminance_hard_cuts_still_split_color_scenes(tmp_path):
    source = create_video(
        tmp_path / "colors.mp4",
        colors=[[0, 0, 255]] * 4 + [[0, 130, 0]] * 4,
        pts=[index * 3000 for index in range(8)],
    )
    assert video.probe_video(source).scene_boundaries == [0, 4, 8]


def test_writer_rejects_missing_extra_or_invalid_frames(tmp_path):
    source = create_video(tmp_path / "source.mp4", pts=[0, 3000])
    info = video.probe_video(source)
    writer = video.VideoWriter(tmp_path / "short.mp4", info)
    writer.write(np.zeros((80, 120, 3), np.uint8))
    with pytest.raises(ValueError, match="Only 1"):
        writer.close()
    assert writer.closed
    writer = video.VideoWriter(tmp_path / "extra.mp4", info)
    for _ in range(2):
        writer.write(np.zeros((80, 120, 3), np.uint8))
    with pytest.raises(ValueError, match="gained"):
        writer.write(np.zeros((80, 120, 3), np.uint8))
    writer.close()
    writer = video.VideoWriter(tmp_path / "invalid.mp4", info)
    with pytest.raises(ValueError, match="invalid"):
        writer.write(np.zeros((80, 120, 3), np.float32))
    writer.close(complete=False)


def test_mux_timeout_is_bounded_and_not_reported_as_success(monkeypatch, tmp_path):
    source = create_video(tmp_path / "source.mp4")

    def fail(command, **kwargs):
        assert kwargs["timeout"] == 3600
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(
        RuntimeError, match="Finishing the video timed out after 3600 seconds"
    ) as failure:
        video.mux_audio(source, source, tmp_path / "output.mp4", video.probe_video(source))
    assert isinstance(failure.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize("stderr", [None, "diagnostic context\nInvalid source media packet"])
def test_mux_error_reports_finishing_stage_status_and_captured_diagnostic(
    monkeypatch, tmp_path, stderr
):
    source = create_video(tmp_path / "source.mp4")

    def fail(command, **kwargs):
        assert kwargs["check"] and kwargs["capture_output"] and kwargs["text"]
        raise subprocess.CalledProcessError(7, command, stderr=stderr)

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(
        RuntimeError, match="Finishing the video failed \\(exit status 7\\)"
    ) as failure:
        video.mux_audio(source, source, tmp_path / "output.mp4", video.probe_video(source))
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    if stderr:
        assert stderr in str(failure.value)


def test_mux_failure_keeps_source_and_never_publishes_partial_video(monkeypatch, tmp_path):
    from watermark_cleaner.core import WatermarkCleaner

    source = create_video(tmp_path / "source.mp4", pts=[0, 3000, 6000], audio=True)
    original = source.read_bytes()

    class IdentityEngine:
        def restore(self, frames, masks, progress=None):
            return frames.copy()

    def fail(command, **kwargs):
        assert "_mux_audio_packets" in command[2]
        Path(command[-3]).write_bytes(b"incomplete mux output")
        raise subprocess.CalledProcessError(1, command, stderr="Native media container failed")

    monkeypatch.setattr(video.subprocess, "run", fail)
    output = tmp_path / "cleaned.mp4"
    progress = []
    with pytest.raises(RuntimeError, match="Native media container failed"):
        WatermarkCleaner(engine=IdentityEngine()).run(
            source,
            output,
            progress_callback=progress.append,
            watermark_regions=[{"bbox": [0.1, 0.1, 0.3, 0.3], "start_frame": 0, "end_frame": 3}],
        )
    assert 91 in progress and 100 not in progress
    assert not output.exists()
    assert not list(tmp_path.glob("temporal-*"))
    assert source.read_bytes() == original


def test_unrepresentable_shared_movie_clock_is_rejected(tmp_path):
    source = create_video(tmp_path / "audio.mp4", audio=True)
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        video._movie_timescale(source, Fraction(1, 2_147_483_647))


def test_mux_keeps_an_existing_destination(tmp_path):
    source = create_video(tmp_path / "source.mp4", audio=True)
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing user content")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        video.mux_audio(source, source, output, video.probe_video(source))
    assert output.read_bytes() == b"existing user content"


def test_verifier_rejects_retimed_same_count_video(tmp_path):
    source = create_video(tmp_path / "source.mp4", pts=[0, 3000, 9000])
    wrong = create_video(tmp_path / "wrong.mp4", pts=[0, 3000, 6000])
    with pytest.raises(ValueError, match="timing"):
        video.verify_output(source, wrong)


def test_audio_loss_is_not_a_successful_verification(tmp_path):
    source = create_video(tmp_path / "audio.mp4", audio=True)
    silent = create_video(tmp_path / "silent.mp4")
    with pytest.raises(ValueError, match="audio"):
        video.verify_output(source, silent)


def test_same_audio_bytes_with_shifted_timestamps_are_rejected(tmp_path):
    source = create_video(tmp_path / "source.mp4", audio=True)
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(
        [
            video.ffmpeg_path(),
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-copyts",
            "-i",
            str(source),
            "-itsoffset",
            "0.25",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            str(shifted),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert video.audio_signatures(source) == video.audio_signatures(shifted)
    with pytest.raises(ValueError, match="audio timing"):
        video.verify_output(source, shifted)


def test_bt2020_conversion_matches_its_output_metadata(tmp_path):
    from dataclasses import replace

    source = create_video(tmp_path / "source.mp4", pts=[0, 3000])
    info = replace(video.probe_video(source), colorspace=9, color_primaries=9, color_trc=16)
    output = tmp_path / "bt2020.mp4"
    writer = video.VideoWriter(output, info)
    writer.stream.options = {**writer.stream.options, "crf": "0"}
    color = np.full((80, 120, 3), [200, 80, 30], np.uint8)
    writer.write(color)
    writer.write(color)
    writer.close()
    assert video.probe_video(output).colorspace == 9
    restored = next(video.iter_frames(output))
    reference = (
        av.VideoFrame.from_ndarray(color, format="bgr24")
        .reformat(format="yuv420p", src_colorspace="BT2020", dst_colorspace="BT2020")
        .reformat(format="bgr24", src_colorspace="BT2020", dst_colorspace="BT2020")
        .to_ndarray()
    )
    assert np.array_equal(restored, reference)


def test_full_range_video_does_not_become_washed_out(tmp_path):
    from dataclasses import replace

    source = create_video(tmp_path / "source.mp4", pts=[0, 3000])
    info = replace(video.probe_video(source), color_range=2)
    writer = video.VideoWriter(tmp_path / "full-range.mp4", info)
    writer.stream.options = {**writer.stream.options, "crf": "0"}
    color = np.zeros((80, 120, 3), np.uint8)
    color[:, 60:] = 255
    writer.write(color)
    writer.write(color)
    writer.close()
    output = video.probe_video(tmp_path / "full-range.mp4")
    restored = next(video.iter_frames(tmp_path / "full-range.mp4"))
    assert output.color_range == 2
    assert restored[:, :50].max() == 0
    assert restored[:, 70:].min() == 255
