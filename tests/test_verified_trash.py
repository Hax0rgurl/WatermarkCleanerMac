from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import pytest

import watermark_cleaner.completion as completion
from watermark_cleaner.completion import (
    OutputVerificationError,
    SourceIdentity,
    SourceIdentityError,
    TrashLocationUnknown,
    TrashMoveError,
    TrashReceipt,
    consume_selection_manifest,
    create_job_workspace,
    mark_job_phase,
    move_to_macos_trash,
    prepare_job_publication,
    recover_stale_workspaces,
    run_verified_cleaning_job,
    verify_cleaned_video,
)

ROOT = Path(__file__).resolve().parents[1]


def _events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _trash_into(directory: Path, call_order: list[str] | None = None):
    directory.mkdir(parents=True, exist_ok=True)

    def trash(path: Path, identity: SourceIdentity) -> TrashReceipt:
        if call_order is not None:
            call_order.append("trash")
        assert SourceIdentity.capture(path) == identity
        destination = directory / path.name
        path.replace(destination)
        return TrashReceipt(
            status="trashed",
            source=str(path),
            trashed=str(destination),
            device=identity.device,
            inode=identity.inode,
        )

    return trash


def test_success_verifies_full_output_before_identity_pinned_trash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected-anywhere.mov"
    output = tmp_path / "output" / "selected-anywhere-cleaned.mp4"
    audit = tmp_path / "cleanup-audit.jsonl"
    source.write_bytes(b"original-video")
    identity = SourceIdentity.capture(source)
    call_order: list[str] = []

    def cleaner(input_path: Path, output_path: Path) -> None:
        call_order.append("clean")
        assert input_path == source
        output_path.write_bytes(b"cleaned-video")

    def verifier(output_path: Path, *, source: Path) -> None:
        call_order.append("verify")
        assert output_path == output
        assert SourceIdentity.capture(source) == identity

    result = run_verified_cleaning_job(
        identity,
        output,
        cleaner,
        verifier=verifier,
        trash=_trash_into(tmp_path / ".Trash", call_order),
        audit_path=audit,
    )

    assert call_order == ["clean", "verify", "trash"]
    assert not source.exists()
    assert output.read_bytes() == b"cleaned-video"
    assert Path(result.trashed).read_bytes() == b"original-video"
    assert _events(audit)[-1]["event"] == "completed_and_trashed"


def test_replacement_during_cleaning_is_never_trashed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(completion, "RUNTIME_ROOT", tmp_path / "runtime")
    source = tmp_path / "source.mp4"
    original_at_new_path = tmp_path / "selected-original-backup.mp4"
    output = tmp_path / "output.mp4"
    audit = tmp_path / "audit.jsonl"
    source.write_bytes(b"selected original")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=tmp_path / "Working",
    )
    trash_called = False

    def cleaner(input_path: Path, output_path: Path) -> None:
        input_path.replace(original_at_new_path)
        input_path.write_bytes(b"replacement at reused path")
        output_path.write_bytes(b"cleaned output")
        prepare_job_publication(workspace, output_path, output_path)

    def trash(_path: Path, _identity: SourceIdentity) -> TrashReceipt:
        nonlocal trash_called
        trash_called = True
        raise AssertionError("changed source must never reach Trash")

    with pytest.raises(SourceIdentityError, match="changed"):
        run_verified_cleaning_job(
            identity,
            output,
            cleaner,
            verifier=lambda *_args, **_kwargs: None,
            trash=trash,
            audit_path=audit,
            workspace=workspace,
        )

    assert not trash_called
    assert source.read_bytes() == b"replacement at reused path"
    assert original_at_new_path.read_bytes() == b"selected original"
    assert not output.exists()
    assert any(event["event"] == "source_identity_changed" for event in _events(audit))


def test_stale_path_reuse_is_rejected_before_cleaning(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"first file")
    identity = SourceIdentity.capture(source)
    source.unlink()
    source.write_bytes(b"new file at same path")
    cleaner_called = False

    def cleaner(_source: Path, _output: Path) -> None:
        nonlocal cleaner_called
        cleaner_called = True

    with pytest.raises(SourceIdentityError, match="changed"):
        run_verified_cleaning_job(identity, output, cleaner)

    assert not cleaner_called
    assert source.read_bytes() == b"new file at same path"


def test_symbolic_link_is_never_accepted_as_source(tmp_path: Path) -> None:
    target = tmp_path / "target.mp4"
    link = tmp_path / "link.mp4"
    target.write_bytes(b"video")
    link.symlink_to(target)

    with pytest.raises(SourceIdentityError, match="symbolic link"):
        SourceIdentity.capture(link)

    assert target.read_bytes() == b"video"


def test_directory_replacement_is_rejected_before_trash(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    identity = SourceIdentity.capture(source)
    source.unlink()
    source.mkdir()

    with pytest.raises(SourceIdentityError, match="non-file"):
        completion.validate_source_identity(identity)

    assert source.is_dir()


def test_processing_failure_quarantines_partial_and_keeps_source(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(completion, "RUNTIME_ROOT", tmp_path / "runtime")
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    audit = tmp_path / "audit.jsonl"
    source.write_bytes(b"original")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=tmp_path / "Working",
    )

    def cleaner(_source: Path, output_path: Path) -> None:
        output_path.write_bytes(b"partial")
        prepare_job_publication(workspace, output_path, output_path)
        raise RuntimeError("engine stopped")

    with pytest.raises(RuntimeError, match="engine stopped"):
        run_verified_cleaning_job(
            identity,
            output,
            cleaner,
            audit_path=audit,
            workspace=workspace,
        )

    assert source.read_bytes() == b"original"
    assert not output.exists()
    quarantined = list((tmp_path / "runtime" / "Failed Jobs").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"


def test_faststart_mp4_truncated_after_first_frame_fails_full_decode(
    tmp_path: Path,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    valid = tmp_path / "valid-faststart.mp4"
    truncated = tmp_path / "truncated-faststart.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=4:size=96x96:rate=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(valid),
        ],
        check=True,
        capture_output=True,
    )
    data = valid.read_bytes()
    truncated.write_bytes(data[: int(len(data) * 0.72)])

    first_frame = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(truncated),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
    )
    assert first_frame.returncode == 0, "fixture must fool first-frame validation"

    with pytest.raises(OutputVerificationError, match="complete cleaned output"):
        verify_cleaned_video(
            truncated,
            ffmpeg_executable=ffmpeg,
        )


def test_full_decode_uses_xerror_with_bounded_timeout(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"fixture")
    captured = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def runner(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    verify_cleaned_video(
        output,
        runner=runner,
        ffmpeg_executable="/test/ffmpeg",
    )

    assert "-xerror" in captured["command"]
    assert "-frames:v" not in captured["command"]
    assert captured["kwargs"]["timeout"] == 3600


def test_output_replacement_during_full_decode_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    original = tmp_path / "output-that-was-verified.mp4"
    output.write_bytes(b"first cleaned file")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def runner(_command, **_kwargs):
        output.replace(original)
        output.write_bytes(b"replacement")
        return Completed()

    with pytest.raises(OutputVerificationError, match="changed during"):
        verify_cleaned_video(
            output,
            runner=runner,
            ffmpeg_executable="/test/ffmpeg",
        )

    assert original.read_bytes() == b"first cleaned file"
    assert output.read_bytes() == b"replacement"


def test_output_symlink_is_rejected_before_full_decode(tmp_path: Path) -> None:
    target = tmp_path / "other-file.mp4"
    output = tmp_path / "output.mp4"
    target.write_bytes(b"not an app-created output")
    output.symlink_to(target)
    runner_called = False

    def runner(_command, **_kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("a symlink must not reach FFmpeg")

    with pytest.raises(OutputVerificationError, match="regular file"):
        verify_cleaned_video(
            output,
            runner=runner,
            ffmpeg_executable="/test/ffmpeg",
        )

    assert not runner_called
    assert target.read_bytes() == b"not an app-created output"


def test_replacement_after_publication_is_rejected_before_trash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "source-cleaned.mp4"
    published_elsewhere = tmp_path / "published-moved-by-user.mp4"
    source.write_bytes(b"selected source")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=tmp_path / "Working",
    )
    trash_called = False

    def cleaner(_source: Path, visible: Path) -> None:
        partial = workspace / "video.partial.mp4"
        partial.write_bytes(b"app-published output")
        prepare_job_publication(workspace, partial, visible)
        os.link(partial, visible)
        partial.unlink()
        visible.replace(published_elsewhere)
        visible.write_bytes(b"replacement at output path")

    def trash(_source: Path, _identity: SourceIdentity) -> TrashReceipt:
        nonlocal trash_called
        trash_called = True
        raise AssertionError("replaced output must never reach Trash")

    with pytest.raises(OutputVerificationError, match="changed after publication"):
        run_verified_cleaning_job(
            identity,
            output,
            cleaner,
            verifier=lambda *_args, **_kwargs: None,
            trash=trash,
            audit_path=tmp_path / "audit.jsonl",
            workspace=workspace,
        )

    assert not trash_called
    assert source.read_bytes() == b"selected source"
    assert output.read_bytes() == b"replacement at output path"
    assert published_elsewhere.read_bytes() == b"app-published output"


def test_nonzero_helper_exit_is_unconfirmed_even_if_path_disappears(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    identity = SourceIdentity.capture(source)
    helper = tmp_path / "trash-item"
    helper.write_text("fixture", encoding="utf-8")
    hidden = tmp_path / "unknown-location"

    class Completed:
        returncode = 74
        stdout = ""
        stderr = "interrupted"

    def runner(_command, **_kwargs):
        source.replace(hidden)
        return Completed()

    with pytest.raises(TrashMoveError, match="did not confirm"):
        move_to_macos_trash(
            source,
            identity,
            runner=runner,
            helper=str(helper),
        )

    assert hidden.read_bytes() == b"original"


def test_location_unknown_is_never_completed_and_trashed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    audit = tmp_path / "audit.jsonl"
    source.write_bytes(b"original")
    identity = SourceIdentity.capture(source)

    def trash(path: Path, expected: SourceIdentity) -> TrashReceipt:
        return TrashReceipt(
            status="location_unknown",
            source=str(path),
            trashed=None,
            device=expected.device,
            inode=expected.inode,
        )

    with pytest.raises(TrashLocationUnknown, match="did not provide"):
        run_verified_cleaning_job(
            identity,
            output,
            lambda _source, result: result.write_bytes(b"cleaned"),
            verifier=lambda *_args, **_kwargs: None,
            trash=trash,
            audit_path=audit,
        )

    events = _events(audit)
    assert events[-1]["event"] == "trash_location_unknown"
    assert all(event["event"] != "completed_and_trashed" for event in events)


def test_post_commit_audit_failure_cannot_reverse_success(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    audit_directory = tmp_path / "audit-is-a-directory"
    audit_directory.mkdir()
    source.write_bytes(b"original")
    identity = SourceIdentity.capture(source)

    result = run_verified_cleaning_job(
        identity,
        output,
        lambda _source, destination: destination.write_bytes(b"cleaned"),
        verifier=lambda *_args, **_kwargs: None,
        trash=_trash_into(tmp_path / ".Trash"),
        audit_path=audit_directory,
    )

    assert not source.exists()
    assert Path(result.trashed).is_file()
    assert output.read_bytes() == b"cleaned"


def test_successful_queue_entry_is_consumed_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    first_identity = SourceIdentity.capture(first)
    second_identity = SourceIdentity.capture(second)
    manifest = tmp_path / "current.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "selected_at": "test",
                "sources": [
                    first_identity.__dict__,
                    second_identity.__dict__,
                ],
            }
        ),
        encoding="utf-8",
    )

    assert consume_selection_manifest(manifest, first_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sources"] == [second_identity.__dict__]


def test_queue_consumption_preserves_unrelated_malformed_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"one")
    identity = SourceIdentity.capture(source)
    malformed = {"path": "/kept-for-diagnosis"}
    manifest = tmp_path / "current.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "sources": [malformed, identity.__dict__],
            }
        ),
        encoding="utf-8",
    )

    assert consume_selection_manifest(manifest, identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sources"] == [malformed]


def test_sigterm_stale_workspace_is_quarantined_on_next_startup(
    tmp_path: Path,
) -> None:
    working = tmp_path / "Working"
    quarantine = tmp_path / "Failed Jobs"
    job = working / "job-sigterm"
    script = (
        "import pathlib,sys,time;"
        "p=pathlib.Path(sys.argv[1]);"
        "p.mkdir(parents=True);"
        "(p/'video.partial.mp4').write_bytes(b'partial');"
        "print('READY', flush=True);"
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(job)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.terminate()
    process.wait(timeout=5)

    recovered = recover_stale_workspaces(
        working_root=working,
        quarantine_root=quarantine,
        audit_path=tmp_path / "audit.jsonl",
    )

    assert not job.exists()
    assert len(recovered) == 1
    assert (recovered[0] / "video.partial.mp4").read_bytes() == b"partial"


def test_sigterm_after_visible_publish_quarantines_exact_uncommitted_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output" / "source-cleaned.mp4"
    working = tmp_path / "Working"
    quarantine = tmp_path / "Failed Jobs"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x48:r=6:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    original = source.read_bytes()
    output.parent.mkdir()
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=working,
    )
    # Use the real decoding, verification, callback and exclusive publication
    # path. Only the costly restoration model is replaced with an identity engine.
    script = """
import pathlib
import sys
import time

from watermark_cleaner.completion import prepare_job_publication
from watermark_cleaner.core import WatermarkCleaner

workspace, source, output = map(pathlib.Path, sys.argv[1:])

class IdentityEngine:
    def restore(self, frames, masks, progress=None):
        return frames.copy()

def progress(percent):
    if percent == 100:
        print('PUBLISHED', flush=True)
        time.sleep(60)

WatermarkCleaner(engine=IdentityEngine()).run(
    source,
    output,
    working_directory=workspace,
    watermark_regions=[{'bbox': [.1, .1, .3, .3], 'source': 'manual'}],
    before_publish=lambda partial, visible: prepare_job_publication(
        workspace, partial, visible
    ),
    progress_callback=progress,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(workspace), str(source), str(output)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 30)[0], "Publication timed out"
        line = process.stdout.readline().strip()
        if line != "PUBLISHED":
            _, errors = process.communicate(timeout=5)
            pytest.fail(f"Cleaner did not reach publication: {line}\n{errors}")
        published = output.read_bytes()
        published_identity = SourceIdentity.capture(output)
        manifest = json.loads((workspace / "job.json").read_text(encoding="utf-8"))
        assert manifest["phase"] == "publish_prepared"
        assert SourceIdentity.from_mapping(manifest["published_output"]) == published_identity
        assert SourceIdentity.capture(source) == identity
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)

    recovered = recover_stale_workspaces(
        working_root=working,
        quarantine_root=quarantine,
        audit_path=tmp_path / "audit.jsonl",
    )

    assert not output.exists()
    assert source.read_bytes() == original
    assert SourceIdentity.capture(source) == identity
    assert not workspace.exists()
    assert len(recovered) == 1
    quarantined_outputs = [
        path
        for path in quarantine.iterdir()
        if path.is_file() and path.name.endswith("source-cleaned.mp4")
    ]
    assert len(quarantined_outputs) == 1
    assert quarantined_outputs[0].read_bytes() == published
    assert quarantined_outputs[0].stat().st_ino == published_identity.inode
    assert any(
        event["event"] == "failed_output_quarantined"
        and event["reason"] == "stale_job_publish_prepared"
        for event in _events(tmp_path / "audit.jsonl")
    )


def test_stale_trash_requested_job_preserves_output_when_source_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    moved_source = tmp_path / "unknown-trash-location.mp4"
    output = tmp_path / "source-cleaned.mp4"
    working = tmp_path / "Working"
    quarantine = tmp_path / "Failed Jobs"
    source.write_bytes(b"selected source")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=working,
    )
    partial = workspace / "video.partial.mp4"
    partial.write_bytes(b"sole cleaned copy")
    completion.prepare_job_publication(workspace, partial, output)
    os.link(partial, output)
    partial.unlink()
    mark_job_phase(workspace, "trash_requested")
    source.replace(moved_source)

    recovered = recover_stale_workspaces(
        working_root=working,
        quarantine_root=quarantine,
        audit_path=tmp_path / "audit.jsonl",
    )

    assert len(recovered) == 1
    assert output.read_bytes() == b"sole cleaned copy"
    assert moved_source.read_bytes() == b"selected source"
    assert any(
        event["event"] == "stale_output_preserved_ambiguous_trash_state"
        for event in _events(tmp_path / "audit.jsonl")
    )


def test_stale_recovery_never_quarantines_replacement_at_output_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "source-cleaned.mp4"
    published_elsewhere = tmp_path / "published-output-moved-by-user.mp4"
    working = tmp_path / "Working"
    quarantine = tmp_path / "Failed Jobs"
    source.write_bytes(b"selected source")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        output,
        working_root=working,
    )
    partial = workspace / "video.partial.mp4"
    partial.write_bytes(b"app-published output")
    prepare_job_publication(workspace, partial, output)
    os.link(partial, output)
    partial.unlink()
    output.replace(published_elsewhere)
    output.write_bytes(b"unrelated replacement")

    recover_stale_workspaces(
        working_root=working,
        quarantine_root=quarantine,
        audit_path=tmp_path / "audit.jsonl",
    )

    assert output.read_bytes() == b"unrelated replacement"
    assert published_elsewhere.read_bytes() == b"app-published output"
    assert any(
        event["event"] == "recorded_output_identity_mismatch"
        for event in _events(tmp_path / "audit.jsonl")
    )


def test_job_workspace_manifest_pins_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    identity = SourceIdentity.capture(source)
    workspace = create_job_workspace(
        identity,
        tmp_path / "output.mp4",
        working_root=tmp_path / "Working",
    )

    payload = json.loads((workspace / "job.json").read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["phase"] == "created"
    assert payload["source"] == identity.__dict__


@pytest.mark.skipif(sys.platform != "darwin", reason="Foundation Trash is macOS-only")
def test_swift_helper_revalidates_identity_and_emits_commit_receipt(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "trash-item"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            "-target",
            "arm64-apple-macos14.0",
            "-module-cache-path",
            str(tmp_path / "ModuleCache"),
            "-framework",
            "Foundation",
            "-o",
            str(helper),
            str(ROOT / "macos/TrashHelper.swift"),
        ],
        check=True,
        capture_output=True,
    )

    def command(path: Path, expected: SourceIdentity) -> list[str]:
        return [
            str(helper),
            str(path),
            str(expected.device),
            str(expected.inode),
            str(expected.file_type),
            str(expected.size),
            str(expected.mtime_ns),
        ]

    replaced = tmp_path / "replaced.mp4"
    selected_original = tmp_path / "selected-original.mp4"
    replaced.write_bytes(b"selected")
    replaced_identity = SourceIdentity.capture(replaced)
    replaced.replace(selected_original)
    replaced.write_bytes(b"replacement")
    mismatch = subprocess.run(
        command(replaced, replaced_identity),
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert replaced.read_bytes() == b"replacement"
    assert selected_original.read_bytes() == b"selected"

    target = tmp_path / "target.mp4"
    link = tmp_path / "link.mp4"
    target.write_bytes(b"target")
    target_identity = SourceIdentity.capture(target)
    link.symlink_to(target)
    symlink_result = subprocess.run(
        command(link, target_identity),
        check=False,
        capture_output=True,
        text=True,
    )
    assert symlink_result.returncode != 0
    assert link.is_symlink()
    assert target.read_bytes() == b"target"

    directory_path = tmp_path / "directory-replacement.mp4"
    directory_path.write_bytes(b"selected")
    directory_identity = SourceIdentity.capture(directory_path)
    directory_path.unlink()
    directory_path.mkdir()
    directory_result = subprocess.run(
        command(directory_path, directory_identity),
        check=False,
        capture_output=True,
        text=True,
    )
    assert directory_result.returncode != 0
    assert directory_path.is_dir()

    successful = tmp_path / "successful.mp4"
    successful.write_bytes(b"selected")
    successful_identity = SourceIdentity.capture(successful)
    committed = subprocess.run(
        command(successful, successful_identity),
        check=False,
        capture_output=True,
        text=True,
    )
    assert committed.returncode == 0
    receipt = json.loads(committed.stdout)
    assert receipt["status"] == "trashed"
    assert receipt["device"] == successful_identity.device
    assert receipt["inode"] == successful_identity.inode
    trashed = Path(receipt["trashed"])
    assert trashed.is_file()
    trashed.replace(successful)
    assert successful.read_bytes() == b"selected"
